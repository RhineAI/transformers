import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
print(sys.path, end='\n\n')

import torch
from safetensors.torch import load_file, save_file
from interpretability.analysis.utils.draw import draw, draw_elementwise, info

# torch.set_printoptions(threshold=float('inf'), sci_mode=False)
torch.set_printoptions(sci_mode=False)

native_input = input


ANALYSIS_NUM = 5
ANALYSIS_MIN_P = 1e-2

DRAW_MODE = True
OUTPUT_DIR = '/data/disk1/guohaoran/transformers/interpretability/analysis/output/0/layers.27.residual.1/'

MODEL_DICT_PATH = '/data/disk1/guohaoran/models/Qwen3-0.6B/model.safetensors'
STATE_DICT_PATH = '/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/state.safetensors'
IMPORTANCE_DICT_PATH = '/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/importance.safetensors'

import torch


def analysis_residual(input, residual, output, importance_output):
    print('residual_input:', list(input.shape))  # [40, 1024]
    print('residual_residual:', list(residual.shape))  # [40, 1024]
    print('residual_output:', list(output.shape))  # [40, 1024]
    print('residual_importance_output:', list(importance_output.shape))  # [40, 1024]
    print()

    input_positive = input >= 0
    residual_positive = residual >= 0

    same_sign = (input_positive == residual_positive)

    output_positive = output >= 0
    input_same_as_output = (input_positive == output_positive)
    residual_same_as_output = (residual_positive == output_positive)

    importance_input = torch.zeros_like(input)
    importance_residual = torch.zeros_like(residual)

    same_sign_mask = same_sign
    if same_sign_mask.any():
        importance_input[same_sign_mask] = (input[same_sign_mask] / output[same_sign_mask]) * importance_output[
            same_sign_mask]
        importance_residual[same_sign_mask] = (residual[same_sign_mask] / output[same_sign_mask]) * importance_output[
            same_sign_mask]

    diff_sign_mask = ~same_sign
    input_gets_importance = diff_sign_mask & input_same_as_output
    residual_gets_importance = diff_sign_mask & residual_same_as_output

    if input_gets_importance.any():
        importance_input[input_gets_importance] = importance_output[input_gets_importance]

    if residual_gets_importance.any():
        importance_residual[residual_gets_importance] = importance_output[residual_gets_importance]

    if DRAW_MODE:
        draw(input, OUTPUT_DIR + 'state/input.jpg', 'GREEN')
        info(input, OUTPUT_DIR + 'state/input.txt', 'residual input state')
        draw(residual, OUTPUT_DIR + 'state/residual.jpg', 'GREEN')
        info(residual, OUTPUT_DIR + 'state/residual.txt', 'residual residual state')
        draw(output, OUTPUT_DIR + 'state/output.jpg', 'GREEN')
        info(output, OUTPUT_DIR + 'state/output.txt', 'residual output state')

        draw(importance_input, OUTPUT_DIR + 'importance/input.jpg', 'GREEN')
        info(importance_input, OUTPUT_DIR + 'importance/input.txt', 'residual importance_input importance')
        draw(importance_residual, OUTPUT_DIR + 'importance/residual.jpg', 'GREEN')
        info(importance_residual, OUTPUT_DIR + 'importance/residual.txt', 'residual importance_residual importance')
        draw(importance_output, OUTPUT_DIR + 'importance/output.jpg', 'GREEN')
        info(importance_output, OUTPUT_DIR + 'importance/output.txt', 'residual importance_output state')

    print(importance_input)
    print(importance_residual)

    return importance_input, importance_residual


if __name__ == '__main__':
    model_dict = load_file(MODEL_DICT_PATH)
    state_dict = load_file(STATE_DICT_PATH)
    importance_dict = load_file(IMPORTANCE_DICT_PATH)

    input = state_dict['model.layers.27.residual.1.input'].to(torch.bfloat16)[0]
    residual = state_dict['model.layers.27.post_attention_layernorm.input'].to(torch.bfloat16)[0]
    output = state_dict['model.norm.input'].to(torch.bfloat16)[0]
    importance_output = importance_dict['model.lm_head.input'].to(torch.bfloat16)

    importance_input, importance_residual = analysis_residual(input, residual, output, importance_output)

    importance_dict['model.layers.27.residual.1.input'] = importance_input.contiguous()
    importance_dict['model.layers.27.residual.1.residual'] = importance_residual.contiguous()

    save_file(importance_dict, IMPORTANCE_DICT_PATH)
    print('Saved to:', IMPORTANCE_DICT_PATH)
