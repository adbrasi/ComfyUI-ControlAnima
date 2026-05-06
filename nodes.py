"""
IC-LoRA (In-Context LoRA) nodes for Anima in ComfyUI.

Key difference from basic temporal concat (AnimaReferenceLatent):
- Per-token timestep: reference frame gets t=0, target frame gets t=sigma
- The model was TRAINED with this asymmetric timestep pattern
- Standard Load LoRA works (IC-LoRA is PEFT standard)

Workflow: Load Checkpoint → Load LoRA → Anima IC-LoRA Apply → KSampler
"""

import math
import torch
import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP


# ============================================================================
# Strength scheduling curves
# ============================================================================

def _schedule_strength(progress, start_pct, end_pct, fade_in, fade_out, curve):
    """Compute IC-LoRA strength at a given sampling progress.

    Args:
        progress: current sampling progress [0, 1] (0=start, 1=done)
        start_pct: start applying at this progress
        end_pct: stop applying at this progress
        fade_in: duration of fade-in ramp (as fraction of total range)
        fade_out: duration of fade-out ramp (as fraction of total range)
        curve: easing function name

    Returns:
        strength in [0, 1]
    """
    # Outside active range
    if progress < start_pct or progress > end_pct:
        return 0.0

    total_range = end_pct - start_pct
    if total_range <= 0:
        return 0.0

    # Normalize progress within [start, end] to [0, 1]
    t = (progress - start_pct) / total_range

    strength = 1.0

    # Fade-in: ramp from 0 to 1 over the first fade_in fraction
    if fade_in > 0 and t < fade_in:
        strength = min(strength, _ease(t / fade_in, curve))

    # Fade-out: ramp from 1 to 0 over the last fade_out fraction
    if fade_out > 0 and t > (1.0 - fade_out):
        fade_t = (1.0 - t) / fade_out
        strength = min(strength, _ease(fade_t, curve))

    return strength


def _ease(t, curve):
    """Apply easing function to t in [0, 1]."""
    t = max(0.0, min(1.0, t))

    if curve == 'linear':
        return t
    elif curve == 'ease_in':
        # Quadratic ease in: slow start, fast end
        return t * t
    elif curve == 'ease_out':
        # Quadratic ease out: fast start, slow end
        return 1.0 - (1.0 - t) ** 2
    elif curve == 'ease_in_out':
        # Smoothstep: slow start, fast middle, slow end
        return t * t * (3.0 - 2.0 * t)
    elif curve == 'cosine':
        # Cosine interpolation: very smooth
        return (1.0 - math.cos(t * math.pi)) / 2.0
    else:
        return t  # fallback to linear


CURVE_OPTIONS = ['linear', 'ease_in', 'ease_out', 'ease_in_out', 'cosine']


def _compute_step_progress(current_sigma, transformer_options):
    """Compute sampling progress from the sampler step index when available."""
    sample_sigmas = transformer_options.get("sample_sigmas", None)
    if sample_sigmas is None:
        return 0.0

    total_steps = len(sample_sigmas) - 1
    if total_steps <= 0:
        return 0.0

    current_sigmas = transformer_options.get("sigmas", None)
    if current_sigmas is not None:
        current_sigma_tensor = current_sigmas[0].to(device=sample_sigmas.device, dtype=sample_sigmas.dtype)
        matches = torch.where(
            torch.isclose(sample_sigmas, current_sigma_tensor, rtol=1e-4)
        )[0]
        if len(matches) > 0:
            return matches[0].item() / total_steps

    diffs = (sample_sigmas - current_sigma).abs()
    return diffs.argmin().item() / total_steps


def _match_batch(tensor, batch_size, name):
    """Expand batch-1 tensors and fail clearly on incompatible batches."""
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1, -1, -1, -1)
    raise ValueError(
        f"[IC-LoRA] {name} batch size {tensor.shape[0]} does not match "
        f"sampling batch size {batch_size}. Use batch 1 or the exact same batch size."
    )


def _prepare_timestep_tables(timesteps, batch_size, target_frames, ref_frames, device, dtype):
    """Return (target timesteps, zero ref timesteps) with shape (B, T)."""
    timesteps = timesteps.to(device=device, dtype=dtype)

    if timesteps.ndim == 0:
        t_target = timesteps.reshape(1, 1).expand(batch_size, target_frames)
    elif timesteps.ndim == 1:
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"[IC-LoRA] timestep batch size {timesteps.shape[0]} does not "
                f"match sampling batch size {batch_size}."
            )
        t_target = timesteps.unsqueeze(1).expand(-1, target_frames)
    elif timesteps.ndim == 2:
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size, -1)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"[IC-LoRA] timestep batch size {timesteps.shape[0]} does not "
                f"match sampling batch size {batch_size}."
            )
        if timesteps.shape[1] == 1 and target_frames > 1:
            t_target = timesteps.expand(-1, target_frames)
        elif timesteps.shape[1] == target_frames:
            t_target = timesteps
        else:
            raise ValueError(
                f"[IC-LoRA] timestep frame count {timesteps.shape[1]} does not "
                f"match target frame count {target_frames}."
            )
    else:
        raise ValueError(f"[IC-LoRA] unsupported timestep shape: {tuple(timesteps.shape)}")

    t_refs = torch.zeros(batch_size, ref_frames, device=device, dtype=dtype)
    return t_target, t_refs


# ============================================================================
# V1 Node (legacy — for LoRAs trained with ic_lora mode)
# ============================================================================

class AnimaICLoRAApply:
    """Apply IC-LoRA conditioning to Anima model.

    Takes a VAE-encoded reference image and patches the model to:
    1. Temporal concat with per-token timestep
    2. Output slice: only return the target frame prediction

    Supports two concat orders:
    - ref_first=False (original): [noise_T0, reference_T1] — ic_lora training
    - ref_first=True (LTX-2 style): [reference_T0, noise_T1] — ic_lora_full training

    Use ComfyUI's standard "Load LoRA" node to load the IC-LoRA weights.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {"tooltip": "VAE-encoded reference image"}),
                "ref_first": ("BOOLEAN", {"default": False,
                                          "tooltip": "False (legacy ic_lora training): [noise T=0, ref T=1]. True (ic_lora_full/V2 training): [ref T=0, noise T=1]. MUST match training order."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Start applying at this % of sampling"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Stop applying at this % of sampling"}),
                "skip_adaln": ("BOOLEAN", {"default": True,
                                           "tooltip": "Skip LoRA on adaln_modulation layers (recommended, matches LTX-2 behavior). Gives smoother strength control."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, ref_latent, ref_first, start_percent, end_percent, skip_adaln):
        model_clone = model.clone()

        # Optionally remove LoRA patches from adaln_modulation layers
        if skip_adaln and hasattr(model_clone, 'patches'):
            keys_to_remove = [k for k in model_clone.patches if 'adaln_modulation' in k]
            for k in keys_to_remove:
                del model_clone.patches[k]
            if keys_to_remove:
                print(f"[IC-LoRA] Skipped {len(keys_to_remove)} adaln_modulation LoRA patches")

        ref_samples = ref_latent["samples"]
        if ref_samples.ndim == 4:
            ref_samples = ref_samples.unsqueeze(2)

        ref_samples = model_clone.model.process_latent_in(ref_samples)

        model_clone.model_options["ic_lora_ref"] = ref_samples
        model_clone.model_options["ic_lora_ref_first"] = ref_first
        model_clone.model_options["ic_lora_start_percent"] = start_percent
        model_clone.model_options["ic_lora_end_percent"] = end_percent
        # V1 uses step scheduling (no fade)
        model_clone.model_options["ic_lora_fade_in"] = 0.0
        model_clone.model_options["ic_lora_fade_out"] = 0.0
        model_clone.model_options["ic_lora_curve"] = 'linear'

        _register_ic_lora_wrapper(model_clone)

        return (model_clone,)


# ============================================================================
# V2 Node (for LoRAs trained with ic_lora_v2 mode)
# ============================================================================

class AnimaICLoRAV2Apply:
    """Apply IC-LoRA V2 conditioning to Anima model.

    Trained with LTX-2 convention: [ref T=0, target T=1] (ref_first=True).
    LoRA trained WITHOUT adaln_modulation and cross_attn — no skip_adaln needed.

    Supports smooth strength scheduling with fade-in/fade-out curves.

    Use ComfyUI's standard "Load LoRA" node to load the IC-LoRA V2 weights.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {"tooltip": "VAE-encoded reference image"}),
                "ref_first": ("BOOLEAN", {"default": True,
                                          "tooltip": "True (V2/LTX-2 training default): [ref T=0, noise T=1]. MUST match training order."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Start applying at this % of sampling"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Stop applying at this % of sampling"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Fade-in duration (fraction of active range). 0=instant on, 0.2=ramp up over first 20%"}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Fade-out duration (fraction of active range). 0=instant off, 0.3=ramp down over last 30%"}),
                "curve": (CURVE_OPTIONS, {"default": "linear",
                                          "tooltip": "Easing curve for fade-in/fade-out. linear=constant rate, ease_in=slow start, ease_out=slow end, ease_in_out=smooth both ends, cosine=very smooth"}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "curve_preview")
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, ref_latent, ref_first, start_percent, end_percent, fade_in, fade_out, curve):
        model_clone = model.clone()

        ref_samples = ref_latent["samples"]
        if ref_samples.ndim == 4:
            ref_samples = ref_samples.unsqueeze(2)

        ref_samples = model_clone.model.process_latent_in(ref_samples)

        # Accumulate refs: if a previous V2 node already added refs, append to the list
        existing_refs = model_clone.model_options.get("ic_lora_refs", [])
        model_clone.model_options["ic_lora_refs"] = existing_refs + [ref_samples]
        model_clone.model_options["ic_lora_ref_first"] = ref_first
        model_clone.model_options["ic_lora_start_percent"] = start_percent
        model_clone.model_options["ic_lora_end_percent"] = end_percent
        model_clone.model_options["ic_lora_fade_in"] = fade_in
        model_clone.model_options["ic_lora_fade_out"] = fade_out
        model_clone.model_options["ic_lora_curve"] = curve

        _register_ic_lora_wrapper(model_clone)

        preview = _render_curve_preview(start_percent, end_percent, fade_in, fade_out, curve)

        return (model_clone, preview)


# ============================================================================
# Curve preview renderer (pure torch, no external deps)
# ============================================================================

def _render_curve_preview(start_pct, end_pct, fade_in, fade_out, curve, width=256, height=128):
    """Render the strength schedule as a small preview image.

    Returns a ComfyUI IMAGE tensor (1, H, W, 3) float32 [0,1].
    """
    img = torch.zeros(height, width, 3)

    # Background: dark gray
    img[:, :, :] = 0.15

    # Grid lines (subtle)
    grid_color = torch.tensor([0.25, 0.25, 0.25])
    for y_frac in [0.25, 0.5, 0.75]:
        y = int((1.0 - y_frac) * (height - 1))
        img[y, :, :] = grid_color
    for x_frac in [0.25, 0.5, 0.75]:
        x = int(x_frac * (width - 1))
        img[:, x, :] = grid_color

    # Active region background (very subtle highlight)
    x_start = int(start_pct * (width - 1))
    x_end = int(end_pct * (width - 1))
    if x_end > x_start:
        img[:, x_start:x_end + 1, :] += 0.05

    # Sample the curve
    curve_color = torch.tensor([0.3, 0.8, 1.0])  # cyan
    fill_color = torch.tensor([0.15, 0.35, 0.45])  # dark cyan fill

    samples = []
    for px in range(width):
        progress = px / (width - 1)
        s = _schedule_strength(progress, start_pct, end_pct, fade_in, fade_out, curve)
        samples.append(s)

    # Draw filled area under curve + curve line
    for px in range(width):
        s = samples[px]
        if s <= 0:
            continue
        curve_y = int((1.0 - s) * (height - 1))
        # Fill under curve
        if curve_y < height - 1:
            img[curve_y + 1:, px, :] = torch.max(img[curve_y + 1:, px, :], fill_color.unsqueeze(0).expand(height - 1 - curve_y, -1))
        # Curve line (2px thick)
        for dy in range(max(0, curve_y - 1), min(height, curve_y + 2)):
            img[dy, px, :] = curve_color

    # Border
    border_color = torch.tensor([0.4, 0.4, 0.4])
    img[0, :, :] = border_color
    img[-1, :, :] = border_color
    img[:, 0, :] = border_color
    img[:, -1, :] = border_color

    # Clamp and return as (1, H, W, 3)
    img = img.clamp(0, 1)
    return img.unsqueeze(0)


# ============================================================================
# V3 Node (clean latents, no resize, ref noise control)
# ============================================================================

class AnimaICLoRAV3Apply:
    """Apply IC-LoRA V3 conditioning to Anima model.

    Key differences from V2:
    - NO latent resize: ref must match target resolution. Encode your ref image
      at the same resolution you're generating. If sizes mismatch, an error is raised.
    - ref_noise: add controlled gaussian noise to the reference latent, making
      the model use it as an approximate guide instead of an exact copy.
      0.0 = exact copy (standard), 0.05 = subtle variation, 0.15 = loose reference.

    Supports multi-ref (chain nodes), fade in/out, easing curves, and curve preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {"tooltip": "VAE-encoded reference image. MUST be same resolution as target (no resize)."}),
                "ref_first": ("BOOLEAN", {"default": True,
                                          "tooltip": "True (V2/LTX-2 training default): [ref T=0, noise T=1]. MUST match training order."}),
                "ref_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                                        "tooltip": "Noise added to reference. 0=exact copy, 0.05=subtle variation, 0.15=loose guide. Higher = more creative freedom."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Start applying at this % of sampling"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Stop applying at this % of sampling"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Fade-in duration (fraction of active range). 0=instant on, 0.2=ramp up over first 20%"}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Fade-out duration (fraction of active range). 0=instant off, 0.3=ramp down over last 30%"}),
                "curve": (CURVE_OPTIONS, {"default": "linear",
                                          "tooltip": "Easing curve for fade-in/fade-out."}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "curve_preview")
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, ref_latent, ref_first, ref_noise, start_percent, end_percent, fade_in, fade_out, curve):
        model_clone = model.clone()

        ref_samples = ref_latent["samples"]
        if ref_samples.ndim == 4:
            ref_samples = ref_samples.unsqueeze(2)

        ref_samples = model_clone.model.process_latent_in(ref_samples)

        # Add controlled noise to reference latent
        if ref_noise > 0:
            noise = torch.randn_like(ref_samples)
            ref_samples = (1 - ref_noise) * ref_samples + ref_noise * noise

        # Accumulate refs
        existing_refs = model_clone.model_options.get("ic_lora_refs", [])
        model_clone.model_options["ic_lora_refs"] = existing_refs + [ref_samples]
        model_clone.model_options["ic_lora_ref_first"] = ref_first
        model_clone.model_options["ic_lora_start_percent"] = start_percent
        model_clone.model_options["ic_lora_end_percent"] = end_percent
        model_clone.model_options["ic_lora_fade_in"] = fade_in
        model_clone.model_options["ic_lora_fade_out"] = fade_out
        model_clone.model_options["ic_lora_curve"] = curve
        model_clone.model_options["ic_lora_strict_size"] = True  # V3: no resize, error on mismatch

        _register_ic_lora_wrapper(model_clone)

        preview = _render_curve_preview(start_percent, end_percent, fade_in, fade_out, curve)

        return (model_clone, preview)


# ============================================================================
# Latent crop/pad utility (avoids bilinear interpolation artifacts)
# ============================================================================

def _crop_or_pad_to(ref, target_h, target_w):
    """Adjust ref latent to match target spatial dims via center crop or edge replication.

    Used for small differences (1-4 latent pixels) caused by VAE rounding.
    No interpolation — preserves the original latent values exactly.

    ref shape: (B, C, T, H, W)
    """
    rh, rw = ref.shape[3], ref.shape[4]

    # Handle height: crop or pad by repeating edge rows
    if rh > target_h:
        crop = (rh - target_h) // 2
        ref = ref[:, :, :, crop:crop + target_h, :]
    elif rh < target_h:
        pad_top = (target_h - rh) // 2
        pad_bottom = target_h - rh - pad_top
        parts = []
        if pad_top > 0:
            parts.append(ref[:, :, :, :1, :].expand(-1, -1, -1, pad_top, -1))
        parts.append(ref)
        if pad_bottom > 0:
            parts.append(ref[:, :, :, -1:, :].expand(-1, -1, -1, pad_bottom, -1))
        ref = torch.cat(parts, dim=3)

    # Handle width: crop or pad by repeating edge columns
    if rw > target_w:
        crop = (rw - target_w) // 2
        ref = ref[:, :, :, :, crop:crop + target_w]
    elif rw < target_w:
        pad_left = (target_w - rw) // 2
        pad_right = target_w - rw - pad_left
        parts = []
        if pad_left > 0:
            parts.append(ref[:, :, :, :, :1].expand(-1, -1, -1, -1, pad_left))
        parts.append(ref)
        if pad_right > 0:
            parts.append(ref[:, :, :, :, -1:].expand(-1, -1, -1, -1, pad_right))
        ref = torch.cat(parts, dim=4)

    return ref


# ============================================================================
# Shared wrapper (used by V1, V2, and V3)
# ============================================================================

def _register_ic_lora_wrapper(model_patcher):
    """Wrapper: temporal concat + per-token timestep for IC-LoRA."""

    _options_ref = model_patcher.model_options

    def ic_lora_wrapper(executor, *args, **kwargs):
        # Support both single ref (V1: ic_lora_ref) and multi-ref (V2: ic_lora_refs)
        ref_list = _options_ref.get("ic_lora_refs", None)
        single_ref = _options_ref.get("ic_lora_ref", None)

        if ref_list is None and single_ref is None:
            return executor(*args, **kwargs)

        # Track progress by sampler step. Sigma-space is non-linear for Anima.
        current_t = args[1].max().item()
        progress = _compute_step_progress(
            current_t, kwargs.get("transformer_options", {})
        )

        start_pct = _options_ref.get("ic_lora_start_percent", 0.0)
        end_pct = _options_ref.get("ic_lora_end_percent", 1.0)
        fade_in = _options_ref.get("ic_lora_fade_in", 0.0)
        fade_out = _options_ref.get("ic_lora_fade_out", 0.0)
        curve = _options_ref.get("ic_lora_curve", "linear")

        strength = _schedule_strength(progress, start_pct, end_pct, fade_in, fade_out, curve)

        if strength <= 0.0:
            return executor(*args, **kwargs)

        # Unpack
        x = args[0]          # (B, C, T, H, W)
        timesteps = args[1]  # (B,) or scalar
        rest_args = list(args[2:])

        # Build list of reference tensors
        if ref_list is not None:
            refs = ref_list
        else:
            refs = [single_ref]

        # Prepare all references: move to device, expand batch, handle spatial dims
        strict_size = _options_ref.get("ic_lora_strict_size", False)
        prepared_refs = []
        for ref in refs:
            ref = ref.to(device=x.device, dtype=x.dtype)
            ref = _match_batch(ref, x.shape[0], "reference latent")
            rh, rw = ref.shape[3], ref.shape[4]
            th, tw = x.shape[3], x.shape[4]
            if rh != th or rw != tw:
                if strict_size:
                    # V3 mode: no bilinear resize. Use center crop/pad for small
                    # differences (VAE rounding), error for large mismatches.
                    max_diff = max(abs(rh - th), abs(rw - tw))
                    if max_diff > 4:
                        ref_px = f"{rw * 8}x{rh * 8}"
                        tgt_px = f"{tw * 8}x{th * 8}"
                        raise ValueError(
                            f"[IC-LoRA V3] Reference latent {rw}x{rh} (image ~{ref_px}) "
                            f"differs too much from target {tw}x{th} (image ~{tgt_px}). "
                            f"Resize your reference IMAGE to match the target resolution "
                            f"BEFORE VAE encoding."
                        )
                    # Small difference (<=4 latent px): center crop/pad without interpolation
                    ref = _crop_or_pad_to(ref, th, tw)
                else:
                    # V1/V2 mode: bilinear resize (legacy)
                    ref = torch.nn.functional.interpolate(
                        ref.squeeze(2),
                        size=(th, tw),
                        mode='bilinear', align_corners=False,
                    ).unsqueeze(2)
            prepared_refs.append(ref)

        # Concatenate all refs along T dimension: (B, C, N_refs, H, W)
        all_refs = torch.cat(prepared_refs, dim=2)
        n_refs = all_refs.shape[2]

        ref_first = _options_ref.get("ic_lora_ref_first", False)
        orig_T = x.shape[2]

        # Per-token timestep: refs get t=0, target gets t=sigma.
        t_target, t_refs = _prepare_timestep_tables(
            timesteps, x.shape[0], orig_T, n_refs, x.device, timesteps.dtype
        )

        if ref_first:
            # [ref1, ref2, ..., refN, noise]
            x_concat = torch.cat([all_refs, x], dim=2)
            timesteps_concat = torch.cat([t_refs, t_target], dim=1)
        else:
            # [noise, ref1, ref2, ..., refN]
            x_concat = torch.cat([x, all_refs], dim=2)
            timesteps_concat = torch.cat([t_target, t_refs], dim=1)

        # Call _forward with modified inputs
        output_with_ref = executor.class_obj._forward(
            x_concat, timesteps_concat, *rest_args, **kwargs
        )

        # Slice to get only the target frame prediction
        if output_with_ref.shape[2] > orig_T:
            if ref_first:
                output_with_ref = output_with_ref[:, :, -orig_T:, :, :]
            else:
                output_with_ref = output_with_ref[:, :, :orig_T, :, :]

        # Apply strength blending
        if strength >= 1.0:
            return output_with_ref

        # Partial strength: blend with unconditional output.
        # Note: output_with_ref uses _forward (bypasses other model wrappers),
        # while output_without_ref uses executor (passes through wrapper chain).
        # This is inherent to the design — calling executor with modified args
        # would cause infinite recursion. Low practical impact since IC-LoRA
        # is rarely combined with other model patches (ControlNet, IPAdapter, etc).
        output_without_ref = executor(*args, **kwargs)
        return torch.lerp(output_without_ref, output_with_ref, strength)

    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "ic_lora_anima",
        ic_lora_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


# ============================================================================
# Node registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "AnimaICLoRAApply": AnimaICLoRAApply,
    "AnimaICLoRAV2Apply": AnimaICLoRAV2Apply,
    "AnimaICLoRAV3Apply": AnimaICLoRAV3Apply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaICLoRAApply": "Anima IC-LoRA Apply",
    "AnimaICLoRAV2Apply": "Anima IC-LoRA V2 Apply",
    "AnimaICLoRAV3Apply": "Anima IC-LoRA V3 Apply",
}
