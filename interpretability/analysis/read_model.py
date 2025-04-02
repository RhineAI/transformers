from safetensors.torch import load_file

path = "/data/disk1/guohaoran/models/Qwen2.5-0.5B-Instruct/model.safetensors"
state_dict = load_file(path)

for k, v in state_dict.items():
    print(k + ':', list(v.shape))

