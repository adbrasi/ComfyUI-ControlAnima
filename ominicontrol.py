"""
OminiControl inference nodes for Anima in ComfyUI.

V1: Basic spatial/subject modes, per-token timestep, KV cache (broken).
V2: Step-based progress, strength scheduling (fade in/out + curves),
    condition strength + noise, curve preview, skip_adaln.

Workflow: Load Checkpoint → Load LoRA → Anima OminiControl V2 Apply → KSampler
"""

import math
import torch
import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP


# ============================================================================
# Step-based progress (fixes non-linear sigma bug)
# ============================================================================

def _compute_step_progress(current_sigma, transformer_options):
    """Compute sampling progress as step_index / total_steps.

    Unlike sigma-based progress (1 - t/t_max), this is LINEAR with respect
    to actual steps. With flow matching + shift + beta57, sigma-space is
    highly non-linear, so sigma-based progress causes start/end_percent
    to behave unpredictably (e.g., 0.05 change cutting 30% of steps).

    Falls back to sigma-based progress if sample_sigmas is unavailable.
    """
    sample_sigmas = transformer_options.get("sample_sigmas", None)
    if sample_sigmas is None:
        return 0.0

    total_steps = len(sample_sigmas) - 1
    if total_steps <= 0:
        return 0.0

    # Find current step by matching sigma in the schedule
    # sample_sigmas is descending: [sigma_max, ..., sigma_min (≈0)]
    # Use torch.where like HunyuanVideo does
    current_sigmas = transformer_options.get("sigmas", None)
    if current_sigmas is not None:
        matches = torch.where(
            torch.isclose(sample_sigmas, current_sigmas[0], rtol=1e-4)
        )[0]
        if len(matches) > 0:
            step_index = matches[0].item()
            return step_index / total_steps

    # Fallback: find closest sigma in the schedule
    diffs = (sample_sigmas - current_sigma).abs()
    step_index = diffs.argmin().item()
    return step_index / total_steps


# ============================================================================
# Strength scheduling (shared with IC-LoRA nodes)
# ============================================================================

CURVE_OPTIONS = ['linear', 'ease_in', 'ease_out', 'ease_in_out', 'cosine']


def _ease(t, curve):
    """Apply easing function to t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if curve == 'linear':
        return t
    elif curve == 'ease_in':
        return t * t
    elif curve == 'ease_out':
        return 1.0 - (1.0 - t) ** 2
    elif curve == 'ease_in_out':
        return t * t * (3.0 - 2.0 * t)
    elif curve == 'cosine':
        return (1.0 - math.cos(t * math.pi)) / 2.0
    return t


def _schedule_strength(progress, start_pct, end_pct, fade_in, fade_out, curve):
    """Compute condition strength at a given sampling progress.

    Returns strength in [0, 1].
    """
    if progress < start_pct or progress > end_pct:
        return 0.0

    total_range = end_pct - start_pct
    if total_range <= 0:
        return 0.0

    t = (progress - start_pct) / total_range
    strength = 1.0

    if fade_in > 0 and t < fade_in:
        strength = min(strength, _ease(t / fade_in, curve))

    if fade_out > 0 and t > (1.0 - fade_out):
        fade_t = (1.0 - t) / fade_out
        strength = min(strength, _ease(fade_t, curve))

    return strength


# ============================================================================
# Curve preview renderer
# ============================================================================

def _render_curve_preview(start_pct, end_pct, fade_in, fade_out, curve,
                          cond_strength=1.0, width=256, height=128):
    """Render the strength schedule as a preview image.

    Returns a ComfyUI IMAGE tensor (1, H, W, 3) float32 [0,1].
    """
    img = torch.full((height, width, 3), 0.15)

    # Grid lines
    grid_color = torch.tensor([0.25, 0.25, 0.25])
    for y_frac in [0.25, 0.5, 0.75]:
        y = int((1.0 - y_frac) * (height - 1))
        img[y, :, :] = grid_color
    for x_frac in [0.25, 0.5, 0.75]:
        x = int(x_frac * (width - 1))
        img[:, x, :] = grid_color

    # Active region highlight
    x_start = int(start_pct * (width - 1))
    x_end = int(end_pct * (width - 1))
    if x_end > x_start:
        img[:, x_start:x_end + 1, :] += 0.05

    # Curve colors
    curve_color = torch.tensor([1.0, 0.6, 0.2])  # orange for OminiControl
    fill_color = torch.tensor([0.45, 0.25, 0.1])

    # Sample and draw
    for px in range(width):
        progress = px / (width - 1)
        s = _schedule_strength(progress, start_pct, end_pct, fade_in, fade_out, curve)
        s *= cond_strength  # apply global strength
        if s <= 0:
            continue
        curve_y = int((1.0 - s) * (height - 1))
        if curve_y < height - 1:
            fill_h = height - 1 - curve_y
            img[curve_y + 1:, px, :] = torch.max(
                img[curve_y + 1:, px, :],
                fill_color.unsqueeze(0).expand(fill_h, -1)
            )
        for dy in range(max(0, curve_y - 1), min(height, curve_y + 2)):
            img[dy, px, :] = curve_color

    # Border
    border_color = torch.tensor([0.4, 0.4, 0.4])
    img[0, :, :] = border_color
    img[-1, :, :] = border_color
    img[:, 0, :] = border_color
    img[:, -1, :] = border_color

    return img.clamp(0, 1).unsqueeze(0)


# ============================================================================
# V1 Node (legacy — kept for backwards compatibility)
# ============================================================================

class AnimaOminiControlApply:
    """Apply OminiControl conditioning to Anima model (V1 — legacy).

    Takes a VAE-encoded condition image and patches the model to:
    1. Temporal concat: [noise_T0, condition_T1]
    2. Per-token timestep: target=sigma, condition=0
    3. RoPE position manipulation (spatial or subject mode)

    Use V2 for better control. This node is kept for workflow compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cond_latent": ("LATENT", {"tooltip": "VAE-encoded condition image (canny, depth, pose, etc.)"}),
                "position_mode": (["spatial", "subject"], {
                    "default": "spatial",
                    "tooltip": "spatial: pixel correspondence (canny/depth/pose). subject: semantic correspondence (identity/style)."
                }),
                "kv_cache": ("BOOLEAN", {"default": False,
                                         "tooltip": "BROKEN — does not work. Kept for compatibility. Use V2 node instead."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Start applying at this % of sampling (sigma-based, non-linear)"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Stop applying at this % of sampling (sigma-based, non-linear)"}),
                "skip_adaln": ("BOOLEAN", {"default": True,
                                           "tooltip": "Skip LoRA on adaln_modulation layers (recommended)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, cond_latent, position_mode, kv_cache, start_percent, end_percent, skip_adaln):
        model_clone = model.clone()

        if skip_adaln and hasattr(model_clone, 'patches'):
            keys_to_remove = [k for k in model_clone.patches if 'adaln_modulation' in k]
            for k in keys_to_remove:
                del model_clone.patches[k]
            if keys_to_remove:
                print(f"[OminiControl] Skipped {len(keys_to_remove)} adaln_modulation LoRA patches")

        cond_samples = cond_latent["samples"]
        if cond_samples.ndim == 4:
            cond_samples = cond_samples.unsqueeze(2)

        cond_samples = model_clone.model.process_latent_in(cond_samples)

        model_clone.model_options["ominicontrol_cond"] = cond_samples
        model_clone.model_options["ominicontrol_position_mode"] = position_mode
        model_clone.model_options["ominicontrol_kv_cache"] = kv_cache
        model_clone.model_options["ominicontrol_start_percent"] = start_percent
        model_clone.model_options["ominicontrol_end_percent"] = end_percent
        # V1 flags
        model_clone.model_options["ominicontrol_version"] = 1
        model_clone.model_options["ominicontrol_strength"] = 1.0
        model_clone.model_options["ominicontrol_fade_in"] = 0.0
        model_clone.model_options["ominicontrol_fade_out"] = 0.0
        model_clone.model_options["ominicontrol_curve"] = 'linear'

        _register_ominicontrol_wrapper(model_clone)

        return (model_clone,)


# ============================================================================
# V2 Node
# ============================================================================

class AnimaOminiControlV2Apply:
    """Apply OminiControl V2 conditioning to Anima model.

    Improvements over V1:
    - Step-based progress (linear, predictable start/end percent)
    - Condition strength (0-1, blends with unconditioned output)
    - Condition noise (loosens adherence to condition)
    - Fade in/out with easing curves
    - Curve preview output
    - Finer slider granularity (0.01 steps)
    - Removed broken KV cache

    Workflow: Load Checkpoint → Load LoRA → Anima OminiControl V2 Apply → KSampler
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cond_latent": ("LATENT", {"tooltip": "VAE-encoded condition image (canny, depth, pose, etc.)"}),
                "position_mode": (["spatial", "subject"], {
                    "default": "spatial",
                    "tooltip": "spatial: pixel correspondence (canny/depth/pose). subject: semantic correspondence (identity/style)."
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Global condition strength. 1.0=full effect, 0.5=half (blended with unconditional), 0.0=off."}),
                "cond_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                                          "tooltip": "Noise added to condition latent. 0=exact, 0.05=subtle variation, 0.15=loose guide. Higher=more creative freedom."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Start applying at this % of steps (step-based, linear)"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Stop applying at this % of steps (step-based, linear)"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Fade-in duration (fraction of active range). 0=instant on, 0.2=ramp up over first 20%."}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Fade-out duration (fraction of active range). 0=instant off, 0.3=ramp down over last 30%."}),
                "curve": (CURVE_OPTIONS, {"default": "linear",
                                          "tooltip": "Easing curve for fade-in/fade-out."}),
                "skip_adaln": ("BOOLEAN", {"default": True,
                                           "tooltip": "Skip LoRA on adaln_modulation layers (recommended — Anima has internal adaln_lora)."}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "curve_preview")
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, cond_latent, position_mode, strength, cond_noise,
              start_percent, end_percent, fade_in, fade_out, curve, skip_adaln):
        model_clone = model.clone()

        # Skip adaln LoRA patches
        if skip_adaln and hasattr(model_clone, 'patches'):
            keys_to_remove = [k for k in model_clone.patches if 'adaln_modulation' in k]
            for k in keys_to_remove:
                del model_clone.patches[k]
            if keys_to_remove:
                print(f"[OminiControl V2] Skipped {len(keys_to_remove)} adaln_modulation LoRA patches")

        cond_samples = cond_latent["samples"]
        if cond_samples.ndim == 4:
            cond_samples = cond_samples.unsqueeze(2)

        cond_samples = model_clone.model.process_latent_in(cond_samples)

        # Add controlled noise to condition latent
        if cond_noise > 0:
            noise = torch.randn_like(cond_samples)
            cond_samples = (1.0 - cond_noise) * cond_samples + cond_noise * noise

        model_clone.model_options["ominicontrol_cond"] = cond_samples
        model_clone.model_options["ominicontrol_position_mode"] = position_mode
        model_clone.model_options["ominicontrol_start_percent"] = start_percent
        model_clone.model_options["ominicontrol_end_percent"] = end_percent
        model_clone.model_options["ominicontrol_strength"] = strength
        model_clone.model_options["ominicontrol_fade_in"] = fade_in
        model_clone.model_options["ominicontrol_fade_out"] = fade_out
        model_clone.model_options["ominicontrol_curve"] = curve
        # V2 flags
        model_clone.model_options["ominicontrol_version"] = 2
        model_clone.model_options["ominicontrol_kv_cache"] = False

        _register_ominicontrol_wrapper(model_clone)

        preview = _render_curve_preview(
            start_percent, end_percent, fade_in, fade_out, curve,
            cond_strength=strength
        )

        return (model_clone, preview)


# ============================================================================
# Shared wrapper (V1 + V2)
# ============================================================================

def _register_ominicontrol_wrapper(model_patcher):
    """Wrapper: temporal concat + per-token timestep + RoPE manipulation + strength scheduling."""

    _options_ref = model_patcher.model_options

    def ominicontrol_wrapper(executor, *args, **kwargs):
        cond_latents = _options_ref.get("ominicontrol_cond", None)
        if cond_latents is None:
            return executor(*args, **kwargs)

        # --- Progress calculation ---
        current_t = args[1].max().item()
        t_options = kwargs.get("transformer_options", {})
        version = _options_ref.get("ominicontrol_version", 1)

        if version >= 2:
            # V2: step-based progress (linear, predictable)
            progress = _compute_step_progress(current_t, t_options)
        else:
            # V1: sigma-based progress (legacy, non-linear)
            sample_sigmas = t_options.get("sample_sigmas", None)
            if sample_sigmas is not None:
                max_sigma = sample_sigmas.max().item()
                progress = 1.0 - (current_t / max_sigma) if max_sigma > 0 else 0.0
            else:
                progress = 0.0

        # --- Strength scheduling ---
        start_pct = _options_ref.get("ominicontrol_start_percent", 0.0)
        end_pct = _options_ref.get("ominicontrol_end_percent", 1.0)
        fade_in = _options_ref.get("ominicontrol_fade_in", 0.0)
        fade_out = _options_ref.get("ominicontrol_fade_out", 0.0)
        curve = _options_ref.get("ominicontrol_curve", "linear")
        global_strength = _options_ref.get("ominicontrol_strength", 1.0)

        schedule_strength = _schedule_strength(progress, start_pct, end_pct, fade_in, fade_out, curve)
        effective_strength = schedule_strength * global_strength

        if effective_strength <= 0.0:
            return executor(*args, **kwargs)

        # --- KV cache (V1 only, broken — kept for compat) ---
        use_kv_cache = _options_ref.get("ominicontrol_kv_cache", False)
        if use_kv_cache:
            # V1 kv_cache: skip condition on steps 2+
            # This is broken by design (loses effect), but kept for V1 compat
            if not hasattr(ominicontrol_wrapper, '_kv_first_done'):
                ominicontrol_wrapper._kv_first_done = False
                ominicontrol_wrapper._kv_last_sigma = -1.0

            if current_t > ominicontrol_wrapper._kv_last_sigma + 0.01:
                ominicontrol_wrapper._kv_first_done = False
            ominicontrol_wrapper._kv_last_sigma = current_t

            if ominicontrol_wrapper._kv_first_done:
                return executor(*args, **kwargs)

        # --- Prepare condition ---
        position_mode = _options_ref.get("ominicontrol_position_mode", "spatial")

        x = args[0]          # (B, C, T, H, W)
        timesteps = args[1]  # (B,) or scalar
        rest_args = list(args[2:])

        cond = cond_latents.to(device=x.device, dtype=x.dtype)
        if cond.shape[0] < x.shape[0]:
            cond = cond.expand(x.shape[0], -1, -1, -1, -1)

        if cond.shape[3] != x.shape[3] or cond.shape[4] != x.shape[4]:
            cond = torch.nn.functional.interpolate(
                cond.squeeze(2), size=(x.shape[3], x.shape[4]),
                mode='bilinear', align_corners=False,
            ).unsqueeze(2)

        # --- Temporal concat: [noise_T0, condition_T1] ---
        x_concat = torch.cat([x, cond], dim=2)
        orig_T = x.shape[2]

        # Per-token timestep: target=sigma, condition=0
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.ndim == 1:
            t_cond = torch.zeros_like(timesteps)
            timesteps_2 = torch.stack([timesteps, t_cond], dim=1)
        else:
            t_cond = torch.zeros(timesteps.shape[0], 1, device=timesteps.device, dtype=timesteps.dtype)
            timesteps_2 = torch.cat([timesteps, t_cond], dim=1)

        # --- RoPE hook for spatial mode ---
        hook_handle = None
        if position_mode == 'spatial':
            hook_handle = _install_spatial_rope_hook(executor.class_obj)

        try:
            output_with_cond = executor.class_obj._forward(
                x_concat, timesteps_2, *rest_args, **kwargs
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # Mark KV cache first step done (V1 compat)
        if use_kv_cache:
            ominicontrol_wrapper._kv_first_done = True

        # Slice to original temporal size
        if output_with_cond.shape[2] > orig_T:
            output_with_cond = output_with_cond[:, :, :orig_T, :, :]

        # --- Strength blending ---
        if effective_strength >= 1.0:
            return output_with_cond

        # Partial strength: blend conditioned with unconditioned output
        output_without_cond = executor(*args, **kwargs)
        return torch.lerp(output_without_cond, output_with_cond, effective_strength)

    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "ominicontrol_anima",
        ominicontrol_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


# ============================================================================
# Spatial RoPE hook
# ============================================================================

def _install_spatial_rope_hook(model):
    """Hook prepare_embedded_sequence to duplicate noise RoPE for condition tokens."""
    original_prepare = model.prepare_embedded_sequence

    def hooked_prepare(x_B_C_T_H_W, fps=None, padding_mask=None):
        x_B_T_H_W_D, rope_emb, extra_pos_emb = original_prepare(x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
        T = x_B_C_T_H_W.shape[2]
        if T > 1 and rope_emb is not None:
            total_seq = rope_emb.shape[0]
            noise_seq_len = total_seq // T
            noise_rope = rope_emb[:noise_seq_len]
            rope_emb = noise_rope.repeat(T, 1, 1, 1)
        return x_B_T_H_W_D, rope_emb, extra_pos_emb

    model.prepare_embedded_sequence = hooked_prepare

    class HookHandle:
        def remove(self):
            model.prepare_embedded_sequence = original_prepare
    return HookHandle()


# ============================================================================
# Node registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "AnimaOminiControlApply": AnimaOminiControlApply,
    "AnimaOminiControlV2Apply": AnimaOminiControlV2Apply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaOminiControlApply": "Anima OminiControl Apply",
    "AnimaOminiControlV2Apply": "Anima OminiControl V2 Apply",
}
