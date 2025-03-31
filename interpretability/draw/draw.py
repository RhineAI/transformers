import torch
from safetensors.torch import load_file
import numpy as np
import cv2
import os

input_path = '/data/guohaoran/guohaoran/transformers/interpretability/record/3/state.safetensors'
# input_path = '/data/guohaoran/guohaoran/models/Qwen2.5-0.5B-Instruct/model.safetensors'
output_dir = '/data/guohaoran/guohaoran/transformers/interpretability/draw/output/state/3'
# output_dir = '/data/guohaoran/guohaoran/transformers/interpretability/draw/output/model'

color_mode = 'BLUE'

from_bgr = np.array([255, 255, 255], dtype=np.float32)
middle_bgr = np.array([128, 128, 128], dtype=np.float32)
to_bgr = np.array([0, 0, 0], dtype=np.float32)

if color_mode == 'GREEN':
    middle_bgr = np.array([66, 156, 47], dtype=np.float32)
elif color_mode == 'BLUE':
    middle_bgr = np.array([166, 92, 57], dtype=np.float32)


def save(image, name):
    cv2.imwrite(os.path.join(output_dir, name + '.jpg'), image)


def draw(n):
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
    # save(higher_colors, 'higher_colors')

    image = lower_colors + higher_colors
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


os.makedirs(output_dir, exist_ok=True)

state_dict = load_file(input_path)
for k, v in state_dict.items():
    shape = v.shape
    if len(shape) < 2:
        v = v.reshape(1, -1)
    elif len(shape) > 2 and shape[0] == 1:
        v = v[0]
    if v.dtype == torch.bfloat16:
        v = v.to(torch.float16)
    n = v.numpy()
    shape = n.shape
    print('Draw:', k, shape)
    image = draw(n)
    save(image, k)

    clip_size = 5000
    if shape[0] > clip_size or shape[1] > clip_size:
        image_clipped = image[:min(shape[0], clip_size), :min(shape[1], clip_size), :]
        save(image_clipped, k + '.clipped')

