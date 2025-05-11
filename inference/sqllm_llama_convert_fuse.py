import torch
import os
import numpy as np
import re
import argparse
import warnings

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

### llama
replacements = {
    'embed_tokens': 'tok_embeddings',
    'self_attn': 'attention',
    'o_proj': 'wo',
    'mlp': 'feed_forward',
    #'gate_proj': 'w1',
    #'up_proj': 'w3',
    'down_proj': 'w2',
    #'input_layernorm': 'attention_norm',
    #'post_attention_layernorm': 'ffn_norm',
    'lm_head': 'output',
    'lookup_table': 'lut'
}

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str)
parser.add_argument('--bitwidth', type=int)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
bitwidth = args.bitwidth
ckpt = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"))

new_dict = {}
for key, value in ckpt.items():
    # Remove 'model.' from the key
    new_key = key.replace('model.', '')

    # Perform the replacements as specified
    for old, new in replacements.items():
        new_key = new_key.replace(old, new)

    # lut2/3/4 -> lut
    if f'lut' in new_key:
        if f'lut{bitwidth}' in new_key:
            new_key = re.sub(r'(?<=lut)[234]', '', new_key)
        else:
            continue

    # Update the new dictionary
    new_dict[new_key] = value

for key in new_dict.keys():
    if (new_dict[key].dtype == torch.bfloat16):
        new_dict[key] = new_dict[key].half()
    if (f'lut{bitwidth}' in key):
        new_dict[key] = new_dict[key].half()
    if ('qweight' in key):
        new_dict[key] = new_dict[key].contiguous()[:bitwidth,:,:]

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
    key_q_qweight = 'layers.'+str(i)+'.attention.q_proj.qweight'
    key_k_qweight = 'layers.'+str(i)+'.attention.k_proj.qweight'
    key_v_qweight = 'layers.'+str(i)+'.attention.v_proj.qweight'
    new_key_qweight = 'layers.'+str(i)+'.attention.wqkv.qweight'

    new_dict[new_key_qweight] = torch.cat((new_dict[key_q_qweight],
                                            new_dict[key_k_qweight],
                                            new_dict[key_v_qweight]), dim=1)

    del(new_dict[key_q_qweight])
    del(new_dict[key_k_qweight])
    del(new_dict[key_v_qweight])


    key_q_lut = 'layers.'+str(i)+'.attention.q_proj.lut'
    key_k_lut = 'layers.'+str(i)+'.attention.k_proj.lut'
    key_v_lut = 'layers.'+str(i)+'.attention.v_proj.lut'
    new_key_lut = 'layers.'+str(i)+'.attention.wqkv.lut'

    new_dict[new_key_lut] = torch.cat((new_dict[key_q_lut],
                                            new_dict[key_k_lut],
                                            new_dict[key_v_lut]), dim=0)

    del(new_dict[key_q_lut])
    del(new_dict[key_k_lut])
    del(new_dict[key_v_lut])
    
    # gate up fusion
    key_gate_qweight = 'layers.'+str(i)+'.feed_forward.gate_proj.qweight'
    key_up_qweight = 'layers.'+str(i)+'.feed_forward.up_proj.qweight'
    new_key_qweight = 'layers.'+str(i)+'.feed_forward.w1w3.qweight'
    new_dict[new_key_qweight] = torch.cat((new_dict[key_gate_qweight],
                                            new_dict[key_up_qweight]), dim=1)

    del(new_dict[key_gate_qweight])
    del(new_dict[key_up_qweight])

    key_gate_lut = 'layers.'+str(i)+'.feed_forward.gate_proj.lut'
    key_up_lut = 'layers.'+str(i)+'.feed_forward.up_proj.lut'
    new_key_lut = 'layers.'+str(i)+'.feed_forward.w1w3.lut'
    new_dict[new_key_lut] = torch.cat((new_dict[key_gate_lut],
                                            new_dict[key_up_lut]), dim=0)
    del(new_dict[key_gate_lut])
    del(new_dict[key_up_lut])

torch.save(new_dict, os.path.join(ckpt_dir, "converted_pytorch_model.bin"))
del(ckpt)
del(new_dict)
