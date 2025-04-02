import json

import torch
from safetensors.torch import load_file

model_dict = load_file('/data/disk1/guohaoran/models/Qwen2.5-0.5B-Instruct/model.safetensors')
state_dict = load_file('/data/guohaoran/guohaoran/transformers/interpretability/record/0/state.safetensors')
tasks = json.load(open('./tasks.json', 'r', encoding='utf-8'))['tasks']

top_k = 5

current_position = []
current_contributes = []

record = []


def without_batch_size(tensor):
    if tensor.shape[0] == 1:
        return tensor[0]
    return tensor


for i, task in enumerate(tasks):
    t = task['task']
    print(f'\nTask {i}: {t}')

    state = without_batch_size(state_dict[task['state']]).to(torch.bfloat16)
    weight = without_batch_size(model_dict[task['weight']]).to(torch.bfloat16)

    print('state:', list(state.shape))
    print('weight:', list(weight.shape))

    if t == 'head':
        logits = state @ weight.T
        max_value, max_index = torch.max(logits[-1], dim=-1)
        k = max_index.item()
        l = state.shape[0]
        print(f"\nMaximum token index: {k},  Value: {max_value.item()}\n")

        state_k = state[k]
        weight_last = weight[-1]

        print('state_final_norm_k:', list(state_k.shape))  # [896]
        print('model_lm_head_last:', list(weight_last.shape))  # [896]

        elementwise_product = state_k * weight_last
        # print(elementwise_product)

        top_values, top_indices = torch.topk(elementwise_product, k=top_k)

        print(f"\nTop contributing weights in model_lm_head:")
        record_position = []
        for idx, val in zip(top_indices, top_values):
            record_position.append([k, idx.item()])
            print(f"index: [{k}, {idx.item()}]  value: {val.item()}")
        record.append({
            'key': task['weight'],
            'position': record_position
        })

        print(f"\nTop contributing state in state_final_norm:")
        position = []
        contributes = []
        for idx, val in zip(top_indices, top_values):
            position.append([l, idx.item()])
            contributes.append(val.item())
            print(f"index: [{l}, {idx.item()}]  value: {val.item()}")
        current_position = position
        current_contributes = contributes

print('\n\nResult:')
print(record)
