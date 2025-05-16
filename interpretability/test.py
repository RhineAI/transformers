import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import os

from transformers.record.record_service import RecordService

user_prompt = '直接给出最终答案，禁止出现原来的式子。36+12='
messages = [
    {"role": "system", "content": 'You are a helpful assistant.'},
    {"role": "user", "content": user_prompt}
]

print(f'[Initialize] Start at: {time.strftime("%H:%M:%S", time.localtime())}')

model_path = "/data/disk1/guohaoran/models/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='cuda:0',
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# print(model)

text = tokenizer.apply_chat_template(
    messages,
    enable_thinking=False,
    tokenize=False,
    add_generation_prompt=True
)
print('\n[Templated]')
print(text.strip().replace('\n\n', '\n'))
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\n[Inference]")
record_service = RecordService()
record_service.reset()

generate_start = time.time()
result = model(
    **inputs,
    pad_token_id=tokenizer.eos_token_id,
)
# print('result:', result)

logits = result.logits  # [33, 151936]
next_token_id = logits[:, -1, :].argmax(dim=-1).item()
print('next_token_id:', next_token_id)
next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
print('next_token:', next_token)

print('\n[Record]')
lines = []
for k, v in record_service.state.items():
    lines.append(k + ': ' + str(list(v.shape)))
print('\n'.join(lines))
# print('\n'.join(lines[:17]) + '\n\n...\n\n' + '\n'.join(lines[-15:]))

saved_path = '/data/disk1/guohaoran/transformers/interpretability/record/Qwen3-0.6B/0'
os.makedirs(saved_path, exist_ok=True)
save_file(record_service.state, os.path.join(saved_path, 'state.safetensors'))
print('\nSaved to:', saved_path)
