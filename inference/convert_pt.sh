#!/bin/bash
set -x

# Execute the Python script
python pt_llama_convert_fuse.py --ckpt_dir ../cache/pretrained/pretrained-(${1}) --model_name meta-llama/$1
