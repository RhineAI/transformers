import torch
from safetensors.torch import load_file
from draw import draw

torch.set_printoptions(
    threshold=float('inf'),
    sci_mode=False
)

ANALYSIS_NUM = 5
ANALYSIS_MIN_P = 1e-2

DRAW_MODE = False
OUTPUT_DIR = '/data/disk1/guohaoran/transformers/interpretability/analysis/output/0/lm_head/'

model_dict = load_file('/data/disk1/guohaoran/models/Qwen3-0.6B/model.safetensors')
state_dict = load_file('/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/state.safetensors')

lm_head_input = state_dict['model.lm_head.input'].to(torch.bfloat16)[0]
lm_head_weight = model_dict['model.embed_tokens.weight'].to(torch.bfloat16).T
storage_logits = state_dict['model.logits'].to(torch.bfloat16)[0]

print('lm_head_input:', list(lm_head_input.shape))  # [40, 1024]
print('lm_head_weight:', list(lm_head_weight.shape))  # [1024, 151936]
print('storage_logits:', list(storage_logits.shape))  # [40, 151936]

logits = lm_head_input @ lm_head_weight   # shape: [40, 1024] · [1024, 151936] = [40, 151936]

max_value, max_index = torch.max(logits[-1], dim=-1)
print(f"\nMaximum token    index: {max_index.item()}  value: {max_value.item()}")

importance = torch.zeros_like(logits)  # shape like logits: [40, 151936]
importance[-1, max_index] = 1


importance_input = torch.zeros_like(lm_head_input)
importance_weight = torch.zeros_like(lm_head_weight)

indices = torch.topk(importance.view(-1), ANALYSIS_NUM)[1]
rn = importance.size(1)
top_position_list = torch.stack((indices // rn, indices % rn), dim=1)

for i in range(top_position_list.shape[0]):
    row, col = top_position_list[i]
    v = importance[row, col]
    if v < ANALYSIS_MIN_P:
        break
    print(f"\nTop {i}    position: [{row.item()}/{importance.shape[0]-1}, {col.item()}/{importance.shape[1]-1}]  value: {v}\n")

    input_line = lm_head_input[row]
    weight_line = lm_head_weight[:, col]
    elementwise = input_line * weight_line
    print('Elementwise detail info (0~30 of 1024)')
    top5 = torch.topk(input_line, 5).values.tolist()
    print(f'\ninput_line:      top5 {top5}')
    print(input_line[:30])
    top5 = torch.topk(weight_line, 5).values.tolist()
    print(f'\nweight_line:      top5 {top5}')
    print(weight_line[:30])
    top5 = torch.topk(elementwise, 5).values.tolist()
    print(f'\nelementwise:      top5 {top5}')
    print(elementwise[:30])

    elementwise_compare = torch.stack([input_line, weight_line, elementwise])

    importance_input[row] += elementwise * v
    importance_weight[:, col] += elementwise * v

    if DRAW_MODE:
        draw(elementwise_compare, OUTPUT_DIR + f'elementwise_{i}.jpg', 'BLUE')

if DRAW_MODE:
    draw(importance, OUTPUT_DIR + 'output.jpg', 'GREEN')
    draw(importance_weight, OUTPUT_DIR + 'weight.jpg', 'BLUE')
    draw(importance_input, OUTPUT_DIR + 'input.jpg', 'GREEN')
