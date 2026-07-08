"""
scripts/underwater_augmentation.py
====================================
Complete underwater image augmentation pipeline for YOLO-DarkWater.

All effects are implemented as standalone, independently callable classes.
Each class accepts a BGR uint8 numpy array and returns a BGR uint8 numpy array.
A UnderwaterAugmentationPipeline composes them probabilistically.

Augmentation rationale:
    The TrashCan dataset was collected in real underwater conditions but cannot
    cover all deployment scenarios (turbidity, depth, lighting). These augmentations
    synthetically expose the model to conditions it may encounter in the field.

Design constraints:
    - Does NOT overwrite original dataset images.
    - Applied as Ultralytics dataset callbacks (per-batch, on-the-fly).
    - Each class is independent and can be used standalone.
    - Probabilities are configurable per class.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual augmentation classes
# ---------------------------------------------------------------------------

class TurbidityHaze:
    """
    Simulates underwater turbidity and depth-dependent haze.

    Blends the image with a blue-green fog layer whose intensity grows
    with a configurable strength parameter. Mimics suspended particles
    (sediment, plankton) that scatter light and reduce contrast.

    Physical basis: Beer-Lambert law — light intensity decays exponentially
    with depth. The fog colour (blue-green dominant) matches the spectral
    transmission window of seawater.
    """

    def __init__(self, intensity: float = 0.4, color: Tuple[int, int, int] = (100, 140, 80)) -> None:
        """
        Args:
            intensity: Blend weight ∈ [0, 1]. Higher = more haze.
            color:     Fog colour as BGR tuple (default: blue-green underwater haze).
        """
        self.intensity = intensity
        self.color = color  # BGR

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        fog_layer = np.full_like(image, self.color[::-1] if len(self.color) == 3 else self.color)
        # Depth-variant fog: stronger at top (farther objects)
        h, w = image.shape[:2]
        depth_map = np.linspace(self.intensity * 0.5, self.intensity, h).reshape(h, 1, 1)
        blended = image * (1 - depth_map) + fog_layer * depth_map
        return np.clip(blended, 0, 255).astype(np.uint8)


class ArtificialBackscatter:
    """
    Simulates artificial LED backscatter from the drone's lights.

    Injects bright Gaussian blobs in a forward-scatter pattern, mimicking
    the reflection of LED light off suspended particles near the camera lens.
    This is a major source of false positives in shallow turbid water.
    """

    def __init__(
        self,
        n_blobs: int = 5,
        max_intensity: float = 0.6,
        max_radius: float = 0.08,
    ) -> None:
        """
        Args:
            n_blobs:       Number of backscatter blobs to inject.
            max_intensity: Peak blob brightness relative to image (0–1).
            max_radius:    Maximum blob radius as fraction of image width.
        """
        self.n_blobs = n_blobs
        self.max_intensity = max_intensity
        self.max_radius = max_radius

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        for _ in range(random.randint(1, self.n_blobs)):
            # Blobs cluster near image centre-bottom (forward-facing LED position)
            cx = int(w * random.gauss(0.5, 0.25))
            cy = int(h * random.gauss(0.75, 0.15))
            cx = np.clip(cx, 0, w - 1)
            cy = np.clip(cy, 0, h - 1)

            radius = int(w * random.uniform(0.02, self.max_radius))
            intensity = random.uniform(0.2, self.max_intensity) * 255

            # Create Gaussian blob on a temporary layer
            blob = np.zeros((h, w), dtype=np.float32)
            cv2.circle(blob, (cx, cy), radius, intensity, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), radius / 2 + 1)

            # Add blob to all channels (white backscatter)
            result += blob[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)


class LowIllumination:
    """
    Simulates low-light underwater conditions.

    Reduces gamma and applies negative brightness offset, mimicking:
    - Deep water (insufficient sunlight penetration)
    - Nighttime operation
    - Camera underexposure in turbid conditions
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.3, 0.7), exposure_range: Tuple[float, float] = (0.4, 0.8)) -> None:
        self.gamma_range = gamma_range
        self.exposure_range = exposure_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        gamma = random.uniform(*self.gamma_range)
        exposure = random.uniform(*self.exposure_range)

        # Gamma darkening via LUT
        lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
        dark = cv2.LUT(image, lut)

        # Exposure scaling
        dark = (dark.astype(np.float32) * exposure).clip(0, 255).astype(np.uint8)
        return dark


class ColorAttenuation:
    """
    Simulates depth-dependent colour attenuation following Beer-Lambert law.

    Red light is absorbed most rapidly (~1–2m), then green (~5–10m), with
    blue penetrating deepest. This creates the characteristic blue-green cast
    of deeper underwater imagery.

    Per-channel exponential decay with configurable depth level.
    """

    def __init__(self, depth_level: float = 0.5) -> None:
        """
        Args:
            depth_level: 0 = surface (minimal attenuation), 1 = deep water.
        """
        self.depth_level = depth_level

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        depth = random.uniform(0.2, self.depth_level)

        img_float = image.astype(np.float32)
        b, g, r = cv2.split(img_float)

        # Attenuation coefficients: red attenuates 3× faster than blue
        r_attn = np.exp(-3.0 * depth)
        g_attn = np.exp(-1.5 * depth)
        b_attn = np.exp(-0.5 * depth)

        r = (r * r_attn).clip(0, 255)
        g = (g * g_attn).clip(0, 255)
        b = (b * b_attn).clip(0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)


class MotionBlur:
    """
    Simulates motion blur from ROV movement or camera shake.

    Applies a directional linear motion blur kernel with a random angle
    and configurable magnitude. Mimics:
    - ROV forward/lateral movement
    - Camera tilt during descent/ascent
    - Propeller-induced vibration
    """

    def __init__(self, max_kernel_size: int = 15) -> None:
        """
        Args:
            max_kernel_size: Maximum blur kernel size in pixels (must be odd).
        """
        self.max_kernel_size = max_kernel_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        # Random kernel size (odd numbers only)
        k = random.choice(range(5, self.max_kernel_size + 1, 2))
        angle = random.uniform(0, 180)

        # Build directional motion blur kernel
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k

        # Rotate kernel to random angle
        rotation_matrix = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (k, k))
        kernel = kernel / (kernel.sum() + 1e-7)

        return cv2.filter2D(image, -1, kernel)


class UnderwaterNoise:
    """
    Simulates combined Poisson + Gaussian sensor noise in low-light conditions.

    Poisson noise models photon shot noise (dominant in low-light).
    Gaussian noise models electronic/thermal sensor noise.
    Together they approximate realistic underwater camera sensor characteristics.
    """

    def __init__(self, gaussian_std_range: Tuple[float, float] = (5, 25)) -> None:
        self.gaussian_std_range = gaussian_std_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        img_float = image.astype(np.float32) / 255.0

        # Poisson noise (shot noise — dominant at low photon counts)
        # Scale controls the photon count level (lower = noisier)
        scale = random.uniform(30, 80)
        poisson = np.random.poisson(img_float * scale) / scale
        poisson = np.clip(poisson, 0, 1)

        # Additive Gaussian noise (thermal/electronic)
        std = random.uniform(*self.gaussian_std_range) / 255.0
        gaussian = np.random.normal(0, std, img_float.shape)

        noisy = np.clip(poisson + gaussian, 0, 1)
        return (noisy * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Composite Pipeline
# ---------------------------------------------------------------------------

class UnderwaterAugmentationPipeline:
    """
    Composable pipeline that applies a random selection of underwater effects.

    Randomly selects 1–3 effects per image (configurable) with per-effect
    probability controls. Designed to be called as an Ultralytics callback.

    Usage:
        pipeline = UnderwaterAugmentationPipeline(p=0.5)
        augmented = pipeline(image_bgr)

    Args:
        p:           Global probability of applying any augmentation (0–1).
        min_effects: Minimum number of effects to apply (when triggered).
        max_effects: Maximum number of effects to apply.
        effect_probs: Per-effect probability weights (relative, not absolute).
                      Higher weight = more likely to be selected.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_effects: int = 1,
        max_effects: int = 3,
        turbidity_p: float = 0.7,
        backscatter_p: float = 0.4,
        low_light_p: float = 0.5,
        color_attn_p: float = 0.6,
        motion_blur_p: float = 0.3,
        noise_p: float = 0.5,
    ) -> None:
        self.p = p
        self.min_effects = min_effects
        self.max_effects = max_effects

        # Named effects with weights for sampling
        self._effects = [
            ("turbidity",    TurbidityHaze(),          turbidity_p),
            ("backscatter",  ArtificialBackscatter(),  backscatter_p),
            ("low_light",    LowIllumination(),        low_light_p),
            ("color_attn",   ColorAttenuation(),       color_attn_p),
            ("motion_blur",  MotionBlur(),             motion_blur_p),
            ("noise",        UnderwaterNoise(),         noise_p),
        ]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply pipeline to a single BGR image."""
        if random.random() > self.p:
            return image  # no augmentation this time

        if image is None or image.size == 0:
            return image

        # Sample a random number of effects
        n = random.randint(self.min_effects, self.max_effects)

        # Weighted sampling without replacement
        names, callables, weights = zip(*self._effects)
        total = sum(weights)
        norm_weights = [w / total for w in weights]

        selected_indices = np.random.choice(
            len(callables), size=min(n, len(callables)), replace=False, p=norm_weights
        )

        result = image.copy()
        for idx in selected_indices:
            result = callables[idx](result)

        return result

    def apply_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply pipeline to a list of images (for batch processing)."""
        return [self(img) for img in images]

    def __repr__(self) -> str:
        effects = [name for name, _, _ in self._effects]
        return f"UnderwaterAugmentationPipeline(p={self.p}, effects={effects})"


# ---------------------------------------------------------------------------
# Ultralytics callback integration
# ---------------------------------------------------------------------------

def build_augmentation_callback(pipeline: UnderwaterAugmentationPipeline):
    """
    Build an Ultralytics-compatible on_train_batch_start callback that applies
    the augmentation pipeline to training batch images.

    Usage:
        aug_pipeline = UnderwaterAugmentationPipeline(p=0.5)
        callback = build_augmentation_callback(aug_pipeline)
        trainer.add_callback('on_train_batch_start', callback)

    Note: Images in Ultralytics batch are torch.Tensor (B, C, H, W) float32 /255.
    This callback converts to numpy, augments, converts back.
    """
    import torch

    def _callback(trainer):
        batch = trainer.batch
        imgs = batch["img"]  # (B, C, H, W), float32, 0-1

        # Convert to numpy BGR uint8
        imgs_np = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        imgs_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs_np]

        # Apply augmentation
        imgs_aug = [pipeline(img) for img in imgs_np]

        # Convert back to tensor
        imgs_aug_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_aug]
        imgs_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in imgs_aug_rgb
        ]).to(imgs.device)

        trainer.batch["img"] = imgs_tensor

    return _callback


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test all augmentations on a synthetic image
    test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    effects = [
        ("TurbidityHaze",       TurbidityHaze()),
        ("ArtificialBackscatter", ArtificialBackscatter()),
        ("LowIllumination",     LowIllumination()),
        ("ColorAttenuation",    ColorAttenuation()),
        ("MotionBlur",          MotionBlur()),
        ("UnderwaterNoise",     UnderwaterNoise()),
    ]

    for name, effect in effects:
        out = effect(test_img)
        assert out.shape == test_img.shape, f"{name}: shape mismatch"
        assert out.dtype == np.uint8, f"{name}: wrong dtype"
        print(f"  ✓ {name}: {test_img.shape} → {out.shape}")

    pipeline = UnderwaterAugmentationPipeline(p=1.0)
    out = pipeline(test_img)
    assert out.shape == test_img.shape
    print(f"  ✓ Pipeline: {test_img.shape} → {out.shape}")
    print("All augmentation effects OK")