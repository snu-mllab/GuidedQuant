#!/bin/bash
set -x

# Create cache/tokens directory if it doesn't exist
mkdir -p ./cache/tokens
cd ./cache/tokens

# Llama-2 (redpajama)
wget https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Llama-2-7b-hf-redpajama_s1024_blk4096.pt
ln -s "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-13b-hf-redpajama_s1024_blk4096.pt"
ln -s "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-70b-hf-redpajama_s1024_blk4096.pt"

# Llama-3 (redpajama)
wget https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Meta-Llama-3-8B-redpajama_s1024_blk4096.pt
ln -s "Meta-Llama-3-8B-redpajama_s1024_blk4096.pt" "Meta-Llama-3-70B-redpajama_s1024_blk4096.pt"


# Llama-2 (wikitext2)
wget https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Llama-2-7b-hf-wikitext2_s128_blk2048.pt
ln -s "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-13b-hf-wikitext2_s128_blk2048.pt"
ln -s "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-70b-hf-wikitext2_s128_blk2048.pt"
