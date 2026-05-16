"""Training augmentation presets for VisDrone YOLOv8 runs."""


PRESETS = {
    "cpu-lite": {
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.08,
        "scale": 0.35,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "fliplr": 0.5,
        "flipud": 0.0,
        "close_mosaic": 5,
    },
    "balanced": {
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "degrees": 2.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "flipud": 0.0,
        "close_mosaic": 10,
    },
    "aggressive": {
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "degrees": 5.0,
        "translate": 0.12,
        "scale": 0.6,
        "shear": 1.0,
        "perspective": 0.0,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "fliplr": 0.5,
        "flipud": 0.0,
        "close_mosaic": 15,
    },
}


def get_preset(name: str) -> dict:
    """Return a copy of the named augmentation preset."""

    try:
        return PRESETS[name].copy()
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown augmentation preset '{name}'. Valid presets: {valid}") from exc