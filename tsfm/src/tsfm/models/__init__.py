from .patch_v0 import V1_QUANTILES, PatchTransformerV0, PatchV0Config

# named architectures selectable from the training CLI (``--model v1``)
MODEL_PRESETS: dict[str, PatchV0Config] = {
    "v0": PatchV0Config(),
    "v1": PatchV0Config(quantiles=V1_QUANTILES),
    "v2": PatchV0Config(quantiles=V1_QUANTILES, max_horizon=1024, future_tokens=True),
}

__all__ = ["MODEL_PRESETS", "V1_QUANTILES", "PatchTransformerV0", "PatchV0Config"]
