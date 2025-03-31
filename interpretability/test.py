import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

user_prompt = '16+28='
messages = [
    {"role": "system", "content": ''},
    {"role": "user", "content": user_prompt}
]

print(f'[Initialize] Start at: {time.strftime("%H:%M:%S", time.localtime())}')

model_path = "/data/disk1/guohaoran/models/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='cuda:0',
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print('text:', text)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("[Inference]")
record_dict = dict()
generate_start = time.time()
result = model(
    **inputs,
    pad_token_id=tokenizer.eos_token_id,
    record_dict=record_dict
)
# print('result:', result)

logits = result.logits
next_token_id = logits[:, -1, :].argmax(dim=-1).item()
print('next_token_id:', next_token_id)
next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
print('next_token:', next_token)

print('\n[Record Dict]')
lines = []
for k, v in record_dict.items():
    lines.append(k + ': ' + str(list(v.shape)))
print('\n'.join(lines[:13]) + '\n\n...\n\n' + '\n'.join(lines[-15:]))

saved_path = '/data/guohaoran/guohaoran/transformers/interpretability/record/state.safetensors'
save_file(record_dict, saved_path)
print('\nSaved to:', saved_path)
