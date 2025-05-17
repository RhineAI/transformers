import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
print(sys.path, end='\n\n')

import torch
from safetensors.torch import load_file, save_file
from interpretability.analysis.utils.draw import draw, draw_elementwise, info


torch.set_printoptions(threshold=float('inf'), sci_mode=False)

ANALYSIS_NUM = 5
ANALYSIS_MIN_P = 1e-2

DRAW_MODE = True
OUTPUT_DIR = '/data/disk1/guohaoran/transformers/interpretability/analysis/output/0/lm_head/'

MODEL_DICT_PATH = '/data/disk1/guohaoran/models/Qwen3-0.6B/model.safetensors'
STATE_DICT_PATH = '/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/state.safetensors'
IMPORTANCE_DICT_PATH = '/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/importance.safetensors'


def analysis_lm_head(input, weight, output, importance_output=None):
    print('lm_head_input:', list(input.shape))  # [40, 1024]
    print('lm_head_weight:', list(weight.shape))  # [1024, 151936]
    print('lm_head_output:', list(output.shape))  # [40, 151936]
    print()
    
    logits = input @ weight   # shape: [40, 1024] · [1024, 151936] = [40, 151936]

    # Only for last layer calculate the correct token for importance 1 other 0
    if importance_output is None:
        importance_output = torch.zeros_like(logits)  # shape like logits: [40, 151936]
        max_value, max_index = torch.max(logits[-1], dim=-1)
        print(f"\nMaximum token    index: {max_index.item()}  value: {max_value.item()}")
        importance_output[-1, max_index] = 1
    
    # Analysis input and weight importance
    importance_input = torch.zeros_like(input)
    importance_weight = torch.zeros_like(weight)
    
    indices = torch.topk(importance_output.view(-1), ANALYSIS_NUM)[1]
    rn = importance_output.size(1)
    top_position_list = torch.stack((indices // rn, indices % rn), dim=1)

    for i in range(top_position_list.shape[0]):
        row, col = top_position_list[i]  # 激活值
        v = importance_output[row, col]  # current
        if v < ANALYSIS_MIN_P:
            break
        print(f"Top {i}    position: [{row.item()}/{importance_output.shape[0]-1}, {col.item()}/{importance_output.shape[1]-1}]  value: {v}")
    
        input_line = input[row]
        weight_line = weight[:, col]
        elementwise = input_line * weight_line
    
        elementwise_compare = torch.stack([input_line, weight_line, elementwise])
    
        current_elementwise = elementwise * v
        importance_input[row] += current_elementwise
        importance_weight[:, col] += current_elementwise
    
        if DRAW_MODE:
            draw_elementwise(elementwise_compare, OUTPUT_DIR + f'elementwise/{i}.jpg', 'BLUE')
    print()

    if DRAW_MODE:
        draw(input, OUTPUT_DIR + 'state/input.jpg', 'GREEN')
        info(input, OUTPUT_DIR + 'state/input.txt', 'lm_head input state')
        draw(weight, OUTPUT_DIR + 'state/weight.jpg', 'BLUE')
        info(weight, OUTPUT_DIR + 'state/weight.txt', 'lm_head weight state')
        draw(output, OUTPUT_DIR + 'state/output.jpg', 'GREEN')
        info(output, OUTPUT_DIR + 'state/output.txt', 'lm_head output state')

        draw(importance_output, OUTPUT_DIR + 'importance/output.jpg', 'GREEN')
        info(importance_output, OUTPUT_DIR + 'importance/output.txt', 'lm_head output importance')
        draw(importance_weight, OUTPUT_DIR + 'importance/weight.jpg', 'BLUE')
        info(importance_weight, OUTPUT_DIR + 'importance/weight.txt', 'lm_head weight importance')
        draw(importance_input, OUTPUT_DIR + 'importance/input.jpg', 'GREEN')
        info(importance_input, OUTPUT_DIR + 'importance/input.txt', 'lm_head input importance')

    return importance_input, importance_weight
    

if __name__ == '__main__':
    model_dict = load_file(MODEL_DICT_PATH)
    state_dict = load_file(STATE_DICT_PATH)
    importance_dict = load_file(IMPORTANCE_DICT_PATH)
    
    lm_head_input = state_dict['model.lm_head.input'].to(torch.bfloat16)[0]
    lm_head_weight = model_dict['model.embed_tokens.weight'].to(torch.bfloat16).T
    lm_head_output = state_dict['model.logits'].to(torch.bfloat16)[0]
    
    importance_input, importance_weight = analysis_lm_head(lm_head_input, lm_head_weight, lm_head_output)

    importance_dict['model.lm_head.input'] = importance_input.contiguous()
    importance_dict['model.embed_tokens.weight'] = importance_weight.contiguous()

    save_file(importance_dict, IMPORTANCE_DICT_PATH)
    print('Saved to:', IMPORTANCE_DICT_PATH)
