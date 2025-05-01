import torch
from safetensors.torch import load_file

model_dict = load_file('/data/disk1/guohaoran/models/Qwen2.5-0.5B-Instruct/model.safetensors')
state_dict = load_file('/data/disk1/guohaoran/transformers/interpretability/record/Qwen2.5-0.5B-Instruct/0/state.safetensors')

state_final_norm = state_dict['state.final_norm'].to(torch.bfloat16)[0]
model_lm_head = model_dict['model.embed_tokens.weight'].to(torch.bfloat16)
state_logits = state_dict['state.logits'].to(torch.bfloat16)[0]
state_last_logit = state_dict['state.last_logits'].to(torch.bfloat16)

print('state_final_norm:', list(state_final_norm.shape))  # [33, 896]
print('model_lm_head:', list(model_lm_head.shape))  # [151936, 896]
print('state_logits:', list(state_logits.shape))  # [33, 151936]
print('state_last_logit:', list(state_last_logit.shape))  # [151936]

l = state_final_norm.shape[0]

logits = state_final_norm @ model_lm_head.T   # shape: [33, 896] · [896, 151936] = [33, 151936]


# import output index: [33, 19]

max_value, max_index = torch.max(logits[-1], dim=-1)
k = max_index.item()
print(f"\nMaximum token index: {k},  Value: {max_value.item()}\n")


state_final_norm_k = state_final_norm[k]
model_lm_head_last = model_lm_head[-1]

print('state_final_norm_k:', list(state_final_norm_k.shape))  # [896]
print('model_lm_head_last:', list(model_lm_head_last.shape))  # [896]

elementwise_product = state_final_norm_k * model_lm_head_last
# print(elementwise_product)


top_values, top_indices = torch.topk(elementwise_product, k=20)

print(f"\nTop contributing weights in model_lm_head:")
for idx, val in zip(top_indices, top_values):
    print(f"Index: [{k}, {idx.item()}], Value: {val.item()}")

print(f"\nTop contributing state in state_final_norm:")
for idx, val in zip(top_indices, top_values):
    print(f"Index: [{l}, {idx.item()}], Value: {val.item()}")
