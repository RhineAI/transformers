from safetensors.torch import load_file

path = "/data/guohaoran/guohaoran/transformers/interpretability/record/0/state.safetensors"
state_dict = load_file(path)

for k, v in state_dict.items():
    print(k + ':', list(v.shape))

# state.logits
# state.layers.23.self_attn.o_proj
