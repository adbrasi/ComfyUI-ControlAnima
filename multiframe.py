"""
Multi-Frame generation for Anima.

Generates N coherent images in a single pass by exploiting
Anima's native T>1 support (3D RoPE + temporal self-attention).

No training needed — the base Anima model handles T>1 natively.
"""

import torch
import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP


class AnimaMultiFrame:
    """Generate N coherent images in one pass.

    Expands noise from T=1 to T=N with independent noise per frame.
    The model's 3D RoPE and temporal self-attention enforce consistency
    across all frames — same identity, style, colors.

    Output is a LATENT with T=N that can be decoded to N images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "num_frames": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1,
                                       "tooltip": "Number of coherent frames to generate"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, num_frames):
        model_clone = model.clone()
        model_clone.model_options["multiframe_n"] = num_frames

        _register_multiframe_wrapper(model_clone)

        return (model_clone,)


class AnimaMultiFrameLatent:
    """Create a multi-frame empty latent (T=N independent noise per frame)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "conditioning/anima"

    def generate(self, width, height, num_frames, batch_size):
        latent_h = height // 8
        latent_w = width // 8
        # T=N frames, each with independent noise
        latent = torch.zeros(batch_size, 16, num_frames, latent_h, latent_w)
        return ({"samples": latent},)


class AnimaSplitFrames:
    """Split a multi-frame LATENT (T=N) into individual images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "split"
    CATEGORY = "conditioning/anima"
    OUTPUT_IS_LIST = (True,)

    def split(self, samples):
        latent = samples["samples"]  # (B, C, T, H, W)
        if latent.ndim == 4:
            # Already T=1, just return as-is
            return ([{"samples": latent}],)

        frames = []
        T = latent.shape[2]
        for t in range(T):
            frame = latent[:, :, t:t+1, :, :]  # keep T dim as 1
            # Squeeze T for standard ComfyUI latent (B, C, H, W)
            frame_4d = frame.squeeze(2)
            frames.append({"samples": frame_4d})

        return (frames,)


def _register_multiframe_wrapper(model_patcher):
    """Wrapper: expand T=1 noise to T=N for multi-frame generation."""

    _options_ref = model_patcher.model_options

    def multiframe_wrapper(executor, *args, **kwargs):
        n_frames = _options_ref.get("multiframe_n", None)

        if n_frames is None or n_frames <= 1:
            return executor(*args, **kwargs)

        x = args[0]  # (B, C, T, H, W)

        # If already T>1, pass through (e.g., user provided multi-frame latent)
        if x.shape[2] >= n_frames:
            return executor(*args, **kwargs)

        # Expand T=1 to T=N with the SAME noise (will be made independent by the sampler's noise)
        # Actually, the KSampler generates noise of same shape as input latent.
        # If input latent is T=N (from AnimaMultiFrameLatent), noise is already T=N.
        # If input is T=1, we need to expand here.
        if x.shape[2] == 1:
            x_expanded = x.repeat(1, 1, n_frames, 1, 1)
            rest_args = list(args[1:])

            output = executor.class_obj._forward(x_expanded, *rest_args, **kwargs)
            return output

        return executor(*args, **kwargs)

    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "multiframe_anima",
        multiframe_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


NODE_CLASS_MAPPINGS = {
    "AnimaMultiFrame": AnimaMultiFrame,
    "AnimaMultiFrameLatent": AnimaMultiFrameLatent,
    "AnimaSplitFrames": AnimaSplitFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaMultiFrame": "Anima Multi-Frame (T=N)",
    "AnimaMultiFrameLatent": "Anima Multi-Frame Empty Latent",
    "AnimaSplitFrames": "Anima Split Frames",
}
