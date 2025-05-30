from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--quantized', action='store_true', help='Use quantized model')
args = parser.parse_args()

if args.quantized:
    model_name = "jusjinuk/Llama-3.1-8B-Instruct-2bit-GuidedQuant-LNQ"
    model = AnyPrecisionForCausalLM.from_quantized(model_name)
else:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)
print(f"Model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

prompt = "Write me a story about Harry, Ron, and Hermione.\n"
chat = [
    {"role": "system", "content": "You are a helpful, creative, and engaging storyteller.\n"},
    {"role": "user", "content": prompt},
]

inputs_text = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)

cache_size = 800

# Compile the decoding phase with static cache
print("~~~~~~~ Compiling model & Warm up ~~~~~~~")
dummy = torch.randint(0, 1, (1, 1)).to(model.device)
new_tokens = cache_size - dummy.shape[1]
output = model.generate(dummy, 
    max_new_tokens=new_tokens, 
    do_sample=True, 
    temperature=1.0,
    top_p=1.0,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=torch.ones_like(dummy),
    cache_implementation="static",
)
print(f"~~~~~~~ Compilation complete ~~~~~~~\n")
print(f"Prompt: {prompt}\n")
input("Press Enter to start generation...")

torch.cuda.synchronize()
start_time = time.time()

new_tokens = cache_size - inputs["input_ids"].shape[1]
output = model.generate(inputs["input_ids"], 
    max_new_tokens=new_tokens, 
    min_new_tokens=new_tokens,
    do_sample=True, 
    pad_token_id=tokenizer.eos_token_id,
    temperature=1.0,
    top_p=1.0,
    attention_mask=torch.ones_like(inputs["input_ids"]),
    cache_implementation="static",
    streamer=streamer
)

torch.cuda.synchronize()
end_time = time.time()

# Calculate generation speed
token_count = len(output[0]) - len(inputs["input_ids"][0])
tokens_per_second = token_count / (end_time - start_time)
ms_per_token = 1 / tokens_per_second * 1000

print(f"\n( Generation speed: {tokens_per_second:.1f} tok/s | Latency: {ms_per_token:.2f} ms/tok )\n")
