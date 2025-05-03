from safetensors.torch import load_file

path = "/data/disk1/guohaoran/transformers/interpretability/record/Qwen2.5-0.5B-Instruct/0/state.safetensors"
state_dict = load_file(path)

for k, v in state_dict.items():
    print(k + ':', list(v.shape))

# state.logits
# state.layers.23.self_attn.o_proj
