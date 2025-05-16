import torch
import os
from safetensors.torch import load_file, save_file
from draw import draw, draw_elementwise, info

torch.set_printoptions(
    threshold=float('inf'),
    sci_mode=False
)

ANALYSIS_NUM = 5
ANALYSIS_MIN_P = 1e-2

SAVED_PATH = '/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/'

DRAW_MODE = True
OUTPUT_DIR = '/data/disk1/guohaoran/transformers/interpretability/analysis/output/0/lm_head/'

model_dict = load_file('/data/disk1/guohaoran/models/Qwen3-0.6B/model.safetensors')
state_dict = load_file('/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/state.safetensors')

lm_head_input = state_dict['model.lm_head.input'].to(torch.bfloat16)[0]
lm_head_weight = model_dict['model.embed_tokens.weight'].to(torch.bfloat16).T
storage_logits = state_dict['model.logits'].to(torch.bfloat16)[0]

print('lm_head_input:', list(lm_head_input.shape))  # [40, 1024]
print('lm_head_weight:', list(lm_head_weight.shape))  # [1024, 151936]
print('storage_logits:', list(storage_logits.shape))  # [40, 151936]

if DRAW_MODE:
    draw(lm_head_input, OUTPUT_DIR + 'state/input.jpg', 'GREEN')
    info(lm_head_input, OUTPUT_DIR + 'state/input.txt', 'lm_head input state')
    draw(lm_head_weight, OUTPUT_DIR + 'state/weight.jpg', 'BLUE')
    info(lm_head_weight, OUTPUT_DIR + 'state/weight.txt', 'lm_head weight state')
    draw(storage_logits, OUTPUT_DIR + 'state/output.jpg', 'GREEN')
    info(storage_logits, OUTPUT_DIR + 'state/output.txt', 'lm_head output state')

logits = lm_head_input @ lm_head_weight   # shape: [40, 1024] · [1024, 151936] = [40, 151936]

# Only for last layer calculate the correct token for importance 1 other 0
importance = torch.zeros_like(logits)  # shape like logits: [40, 151936]
max_value, max_index = torch.max(logits[-1], dim=-1)
print(f"\nMaximum token    index: {max_index.item()}  value: {max_value.item()}")
importance[-1, max_index] = 1

# Analysis input and weight importance
importance_input = torch.zeros_like(lm_head_input)
importance_weight = torch.zeros_like(lm_head_weight)

indices = torch.topk(importance.view(-1), ANALYSIS_NUM)[1]
rn = importance.size(1)
top_position_list = torch.stack((indices // rn, indices % rn), dim=1)

for i in range(top_position_list.shape[0]):
    row, col = top_position_list[i]  # 激活值
    v = importance[row, col]  # current
    if v < ANALYSIS_MIN_P:
        break
    print(f"\nTop {i}    position: [{row.item()}/{importance.shape[0]-1}, {col.item()}/{importance.shape[1]-1}]  value: {v}")

    input_line = lm_head_input[row]
    weight_line = lm_head_weight[:, col]
    elementwise = input_line * weight_line

    elementwise_compare = torch.stack([input_line, weight_line, elementwise])

    current_elementwise = elementwise * v
    importance_input[row] += current_elementwise
    importance_weight[:, col] += current_elementwise

    if DRAW_MODE:
        draw_elementwise(elementwise_compare, OUTPUT_DIR + f'elementwise/{i}.jpg', 'BLUE')
print()

if DRAW_MODE:
    draw(importance, OUTPUT_DIR + 'importance/output.jpg', 'GREEN')
    info(importance, OUTPUT_DIR + 'importance/output.txt', 'lm_head output importance')
    draw(importance_weight, OUTPUT_DIR + 'importance/weight.jpg', 'BLUE')
    info(importance_weight, OUTPUT_DIR + 'importance/weight.txt', 'lm_head weight importance')
    draw(importance_input, OUTPUT_DIR + 'importance/input.jpg', 'GREEN')
    info(importance_input, OUTPUT_DIR + 'importance/input.txt', 'lm_head input importance')


importance_dict = {
    'model.lm_head.input': importance_input.contiguous(),
    'model.embed_tokens.weight': importance_weight.contiguous(),
}
os.makedirs(SAVED_PATH, exist_ok=True)
save_file(importance_dict, os.path.join(SAVED_PATH, 'importance.safetensors'))
print('Saved to:', SAVED_PATH)
