from safetensors.torch import load_file

path = "/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0/state.safetensors"
state_dict = load_file(path)

for k, v in state_dict.items():
    print(k + ':', list(v.shape))

# state.logits
# state.layers.23.self_attn.o_proj
