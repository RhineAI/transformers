import torch
from safetensors.torch import load_file

model_dict = load_file('/data/disk1/guohaoran/models/Qwen3-0.6B/model.safetensors')
state_dict = load_file('/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/state.safetensors')

lm_head_input = state_dict['model.lm_head.input'].to(torch.bfloat16)[0]
lm_head_weight = model_dict['model.embed_tokens.weight'].to(torch.bfloat16)
state_logits = state_dict['model.logits'].to(torch.bfloat16)[0]

print('state_final_norm:', list(lm_head_input.shape))  # [40, 1024]
print('model_lm_head:', list(lm_head_weight.shape))  # [151936, 1024]
print('state_logits:', list(state_logits.shape))  # [40, 151936]

l = lm_head_input.shape[0]

logits = lm_head_input @ lm_head_weight.T   # shape: [40, 1024] · [1024, 151936] = [40, 151936]


max_value, max_index = torch.max(logits[-1], dim=-1)
k = max_index.item()
print(f"\nMaximum token index: {k},  Value: {max_value.item()}\n")


lm_head_input_k = lm_head_input[k]
lm_head_weight_last = lm_head_weight[-1]

print('state_final_norm_k:', list(lm_head_input_k.shape))  # [1024]
print('model_lm_head_last:', list(lm_head_weight_last.shape))  # [1024]

elementwise_product = lm_head_input_k * lm_head_weight_last
# print(elementwise_product)


top_values, top_indices = torch.topk(elementwise_product, k=20)

print(f"\nTop contributing weights in model_lm_head:")
for idx, val in zip(top_indices, top_values):
    print(f"Index: [{k}, {idx.item()}], Value: {val.item()}")

print(f"\nTop contributing state in state_final_norm:")
for idx, val in zip(top_indices, top_values):
    print(f"Index: [{l}, {idx.item()}], Value: {val.item()}")
