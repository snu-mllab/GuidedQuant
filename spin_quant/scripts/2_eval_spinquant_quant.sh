set -x

bash scripts/2_eval_ptq_save_7b.sh meta-llama/Llama-2-7b-hf 4 4 4 "rotation/7B_W16A4KV4_lr_1.5_seed_0/R.bin"
bash scripts/2_eval_ptq_save_7b.sh meta-llama/Llama-2-7b-hf 4 4 16 "rotation/7B_W16A4KV16_lr_1.5_seed_0/R.bin"

bash scripts/2_eval_ptq_save_13b.sh meta-llama/Llama-2-13b-hf 4 4 4 "rotation/13B_W16A4KV4_lr_1.5_seed_0/R.bin"
bash scripts/2_eval_ptq_save_13b.sh meta-llama/Llama-2-13b-hf 4 4 16 "rotation/13B_W16A4KV16_lr_1.5_seed_0/R.bin"

bash scripts/2_eval_ptq_save_70b.sh meta-llama/Llama-2-70b-hf 4 4 4 "rotation/70B_W16A4KV4_lr_1.5_seed_0/R.bin"
bash scripts/2_eval_ptq_save_70b.sh meta-llama/Llama-2-70b-hf 4 4 16 "rotation/70B_W16A4KV16_lr_1.5_seed_0/R.bin"
