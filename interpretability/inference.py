import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


user_prompt = '直接给出最终答案，一个数字。16+28='
messages = [
    {"role": "system", "content": 'You are a helpful assistant.'},
    {"role": "user", "content": user_prompt}
]

start_time = time.time()
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

print("\n[Inference]")
generate_start = time.time()

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

generated = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=streamer,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
    return_legacy_cache=True,
)

generate_duration = time.time() - generate_start
total_new_tokens = generated.sequences[0].size(0) - inputs.input_ids.size(1)
tpm = total_new_tokens / (generate_duration / 60)

print(f"\n[Status] Generation completed in {generate_duration:.2f}s")
print(f"[Metric] TPM: {tpm:.0f} | Tokens: {total_new_tokens}")
print(f"[Total] Duration: {time.time() - start_time:.2f}s")

