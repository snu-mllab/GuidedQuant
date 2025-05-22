from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import torch
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str)
parser.add_argument('--model_name', type=str)
args = parser.parse_args()


ckpt_dir = args.ckpt_dir
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

model_name = args.model_name

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(ckpt_dir, safe_serialization=False, max_shard_size="160GB")
tokenizer.save_pretrained(ckpt_dir)


### llama
replacements = {
    'embed_tokens': 'tok_embeddings',
    'self_attn': 'attention',
    'o_proj': 'wo',
    'mlp': 'feed_forward',
    'down_proj': 'w2',
    'lm_head': 'output'
}

ckpt = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"))

new_dict = {}
for key, value in ckpt.items():
    # Remove 'model.' from the key
    new_key = key.replace('model.', '')

    # Perform the replacements as specified
    for old, new in replacements.items():
        new_key = new_key.replace(old, new)

    # Update the new dictionary
    new_dict[new_key] = value

for key in new_dict.keys():
    if (new_dict[key].dtype == torch.bfloat16):
        new_dict[key] = new_dict[key].half()
    if ('lut' in key):
        new_dict[key] = new_dict[key].half()
    if ('qweight' in key):
        new_dict[key] = new_dict[key].contiguous()

if "Llama-2-7b" in ckpt_dir:
    layer_num = 32
elif "Llama-2-13b" in ckpt_dir:
    layer_num = 40
elif "Llama-2-70b" in ckpt_dir:
    layer_num = 80
else:
    raise ValueError(f"Unsupported model: {ckpt_dir}")

for i in range(layer_num):
    # qkv fusion
    key_q_weight = 'layers.'+str(i)+'.attention.q_proj.weight'
    key_k_weight = 'layers.'+str(i)+'.attention.k_proj.weight'
    key_v_weight = 'layers.'+str(i)+'.attention.v_proj.weight'
    new_key_weight = 'layers.'+str(i)+'.attention.wqkv.weight'


    new_dict[new_key_weight] = torch.cat((new_dict[key_q_weight],
                                           new_dict[key_k_weight],
                                           new_dict[key_v_weight]), dim=0)

    del(new_dict[key_q_weight])
    del(new_dict[key_k_weight])
    del(new_dict[key_v_weight])
    
    # gate up fusion
    key_gate_weight = 'layers.'+str(i)+'.feed_forward.gate_proj.weight'
    key_up_weight = 'layers.'+str(i)+'.feed_forward.up_proj.weight'
    new_key_weight = 'layers.'+str(i)+'.feed_forward.w1w3.weight'


    new_dict[new_key_weight] = torch.cat((new_dict[key_gate_weight],
                                           new_dict[key_up_weight]), dim=0)

    del(new_dict[key_gate_weight])
    del(new_dict[key_up_weight])

for key in sorted(new_dict.keys()):
    print(key, new_dict[key].shape)

torch.save(new_dict, os.path.join(ckpt_dir, "converted_pytorch_model.bin"))
del(ckpt)
del(new_dict)

