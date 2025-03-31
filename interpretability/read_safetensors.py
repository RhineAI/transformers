from safetensors.torch import load_file

path = "/data/disk1/guohaoran/models/Qwen2.5-0.5B-Instruct/model.safetensors"
state_dict = load_file(path)

print(state_dict.keys())

embed_tokens_weight = state_dict["model.embed_tokens.weight"]
print(embed_tokens_weight.shape)
print(embed_tokens_weight)
