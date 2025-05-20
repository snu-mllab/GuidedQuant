set -x
MODEL_NAME=$1
BITS=$2
NUM_GROUPS=$3
python layerwise_nuq.py $MODEL_NAME --seed_precision $BITS --dataset redpajama --seq_len 4096 --num_examples 1024 --num_groups $NUM_GROUPS
