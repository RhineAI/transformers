import torch
from safetensors.torch import load_file
from draw import draw, draw_elementwise, info

torch.set_printoptions(
    threshold=float('inf'),
    sci_mode=False
)

ANALYSIS_NUM = 5
ANALYSIS_MIN_P = 1e-2

DRAW_MODE = True
OUTPUT_DIR = '/data/disk1/guohaoran/transformers/interpretability/analysis/output/0-28/lm_head/'

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

max_value, max_index = torch.max(logits[-1], dim=-1)
print(f"\nMaximum token    index: {max_index.item()}  value: {max_value.item()}")

importance = torch.zeros_like(logits)  # shape like logits: [40, 151936]
# importance[-1, max_index] = 1
importance[-1, 28] = 1


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

    elementwise_compare = torch.stack([input_line, weight_line, elementwise])

    importance_input[row] += elementwise * v
    importance_weight[:, col] += elementwise * v

    if DRAW_MODE:
        draw_elementwise(elementwise_compare, OUTPUT_DIR + f'elementwise/{i}.jpg', 'BLUE')

if DRAW_MODE:
    draw(importance, OUTPUT_DIR + 'importance/output.jpg', 'GREEN')
    info(importance, OUTPUT_DIR + 'importance/output.txt', 'lm_head output importance')
    draw(importance_weight, OUTPUT_DIR + 'importance/weight.jpg', 'BLUE')
    info(importance_weight, OUTPUT_DIR + 'importance/weight.txt', 'lm_head weight importance')
    draw(importance_input, OUTPUT_DIR + 'importance/input.jpg', 'GREEN')
    info(importance_input, OUTPUT_DIR + 'importance/input.txt', 'lm_head input importance')
