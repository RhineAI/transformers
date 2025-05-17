import os
import cv2
import torch
import numpy as np


CONTEXT_LENGTH = 40
CONTEXT_RANGE = [0, 40]

HIDDEN_SIZE = 1024
HIDDEN_RANGE = [0, 128]

VOCAB_SIZE = 151936
VOCAB_RANGE = [0, 256]

HIDDEN_STATE_DIRECTION = 'VERTICAL'

INFO_NUM = 5


def draw(tensor, path, color_mode='BLUE'):
    image = generate(tensor, color_mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, direction(image.cpu().numpy()))


def info(tensor, path, name):
    torch.set_printoptions(threshold=1000, edgeitems=3, linewidth=120, sci_mode=False)

    flat = tensor.flatten()
    top5 = torch.topk(flat, 5).values
    bottom5 = torch.topk(-flat, 5).values.neg()
    sorted_tensor = flat.sort().values
    n = len(sorted_tensor)
    mid5 = sorted_tensor[n // 2 - 2: n // 2 + 3] if n >= 5 else sorted_tensor

    mean = tensor.mean()
    std = tensor.std()

    text = (
        f"{name}\n\n"
        f"Shape: {list(tensor.shape)}\n\n"
        f"Mean: {mean.item():.6f}\n"
        f"Std Dev: {std.item():.6f}\n\n"
        f"Top 5 values: {top5.tolist()}\n"
        f"Bottom 5 values: {bottom5.tolist()}\n"
        f"Middle 5 values: {mid5.tolist()}\n\n"
        f"{tensor}\n"
    )

    with open(path, mode='w', encoding='utf-8') as f:
        f.write(text)


def direction(image):
    clipped_hidden_size = HIDDEN_RANGE[1] - HIDDEN_RANGE[0]
    if HIDDEN_STATE_DIRECTION == 'VERTICAL' and image.shape[1] == clipped_hidden_size:
        image = np.transpose(image, (1, 0, 2))
    if HIDDEN_STATE_DIRECTION == 'HORIZONTAL' and image.shape[0] == clipped_hidden_size:
        image = np.transpose(image, (1, 0, 2))
    return image


def draw_elementwise(tensor, path, color_mode='BLUE'):
    image_0 = generate(tensor[0], color_mode).cpu().numpy()
    image_1 = generate(tensor[1], color_mode).cpu().numpy()
    image_2 = generate(tensor[2], color_mode).cpu().numpy()
    white = np.full(image_0.shape, 255)
    image = np.array([image_0, white, image_1, white, image_2])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, direction(image))


def generate(tensor, color_mode='BLUE'):
    from_bgr = torch.tensor([255, 255, 255], dtype=torch.float32)
    middle_bgr = torch.tensor([128, 128, 128], dtype=torch.float32)
    to_bgr = torch.tensor([0, 0, 0], dtype=torch.float32)

    if color_mode == 'GREEN':
        middle_bgr = torch.tensor([66, 156, 47], dtype=torch.float32)
    elif color_mode == 'BLUE':
        middle_bgr = torch.tensor([166, 92, 57], dtype=torch.float32)

    t = tensor.to(torch.float32)
    max_val = t.max()
    min_val = t.min()
    rng = max_val - min_val

    if rng == 0:
        s = torch.full_like(t, 0.5)
    else:
        s = (t - min_val) / rng

    mask_lower = s <= 0.5
    mask_higher = s > 0.5

    lower_weights = torch.clamp(2 * s, 0.0, 1.0) * mask_lower
    higher_weights = torch.clamp(2 * (s - 0.5), 0.0, 1.0) * mask_higher

    # Expand to shape [..., 3]
    lower_weights_3d = lower_weights.unsqueeze(-1)
    higher_weights_3d = higher_weights.unsqueeze(-1)

    lower_colors = (1 - lower_weights_3d) * from_bgr + lower_weights_3d * middle_bgr
    lower_colors *= mask_lower.unsqueeze(-1)

    higher_colors = (1 - higher_weights_3d) * middle_bgr + higher_weights_3d * to_bgr
    higher_colors *= mask_higher.unsqueeze(-1)

    image = lower_colors + higher_colors
    image = torch.clamp(image, 0, 255).to(torch.uint8)

    clipped_image = clip(image)
    return clipped_image


def clip(tensor):
    if len(tensor.shape) == 2:
        h = 1
        w = tensor.shape[0]
    else:
        h, w = tensor.shape[:2]

    if w == CONTEXT_LENGTH:
        w_range = CONTEXT_RANGE
    elif w == HIDDEN_SIZE:
        w_range = HIDDEN_RANGE
    elif w == VOCAB_SIZE:
        w_range = VOCAB_RANGE
    else:
        w_range = [0, w]

    if h == CONTEXT_LENGTH:
        h_range = CONTEXT_RANGE
    elif h == HIDDEN_SIZE:
        h_range = HIDDEN_RANGE
    elif h == VOCAB_SIZE:
        h_range = VOCAB_RANGE
    else:
        h_range = [0, h]

    if len(tensor.shape) == 2:
        clipped = tensor[w_range[0]:w_range[1], :]
    else:
        clipped = tensor[h_range[0]:h_range[1], w_range[0]:w_range[1], :]
    return clipped


