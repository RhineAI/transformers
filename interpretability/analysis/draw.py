import os
import cv2
import torch
import numpy as np


clip_size = 5000


def draw(tensor, path, color_mode='BLUE'):
    from_bgr = np.array([255, 255, 255], dtype=np.float32)
    middle_bgr = np.array([128, 128, 128], dtype=np.float32)
    to_bgr = np.array([0, 0, 0], dtype=np.float32)

    if color_mode == 'GREEN':
        middle_bgr = np.array([66, 156, 47], dtype=np.float32)
    elif color_mode == 'BLUE':
        middle_bgr = np.array([166, 92, 57], dtype=np.float32)

    n = tensor.to(torch.float32).numpy()

    max_val, min_val = np.max(n), np.min(n)
    rng = max_val - min_val
    if rng == 0:
        s = np.full_like(n, 0.5)
    else:
        s = (n - min_val) / rng

    mask_lower = s <= 0.5
    mask_higher = s > 0.5

    lower_weights = 2 * s
    lower_weights = np.clip(lower_weights, 0.0, 1.0)
    lower_weights *= mask_lower.astype(np.float32)
    lower_colors = (1 - lower_weights[..., np.newaxis]) * from_bgr + lower_weights[..., np.newaxis] * middle_bgr
    lower_colors *= np.repeat(mask_lower[..., np.newaxis], 3, axis=-1)
    # save(lower_colors, 'lower_colors')

    higher_weights = 2 * (s - 0.5)
    higher_weights = np.clip(higher_weights, 0.0, 1.0)
    higher_weights *= mask_higher.astype(np.float32)
    higher_colors = (1 - higher_weights[..., np.newaxis]) * middle_bgr + higher_weights[..., np.newaxis] * to_bgr
    higher_colors *= np.repeat(mask_higher[..., np.newaxis], 3, axis=-1)

    image = lower_colors + higher_colors
    image = np.clip(image, 0, 255).astype(np.uint8)

    shape = n.shape
    if shape[0] > clip_size or shape[1] > clip_size:
        image = image[:min(shape[0], clip_size), :min(shape[1], clip_size), :]


    cv2.imwrite(path, image)

