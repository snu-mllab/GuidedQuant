#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 -v <version: 2|3> -s <size: 7b|8b|13b|70b> -b <bitwidth: 2|3|4> -m <method: sq|lnq|glnq> [-g <groupsize: 1|2|4>]"
  exit 1
}

# Parse options
while getopts "v:s:b:g:m:" opt; do
  case ${opt} in
    v) VERSION=$OPTARG ;;
    s) SIZE=$OPTARG ;;
    b) BITWIDTH=$OPTARG ;;
    g) GROUPSIZE=$OPTARG ;;
    m) METHOD=$OPTARG ;;
    *) usage ;;
  esac
done

# Validate required arguments
if [[ -z "$VERSION" || -z "$SIZE" || -z "$BITWIDTH" || -z "$METHOD" ]]; then
  usage
fi

# Assign default GROUPSIZE if needed
if [[ "$METHOD" == "lnq" && -z "$GROUPSIZE" ]]; then
  GROUPSIZE=1
fi

# Check if GROUPSIZE is required for selected METHOD
if [[ "$METHOD" == "glnq" && -z "$GROUPSIZE" ]]; then
  echo "Error: -g <groupsize> is required for method '$METHOD'"
  usage
fi

# Determine model prefix
if [[ "$VERSION" == "2" ]]; then
  MODEL_PREFIX="Llama-${VERSION}-${SIZE}-hf"
elif [[ "$VERSION" == "3" ]]; then
  MODEL_PREFIX="Meta-Llama-${VERSION}-${SIZE}"
else
  echo "Unsupported version: $VERSION"
  exit 1
fi

# Set CKPT_DIR based on METHOD
case "$METHOD" in
  sq)
    CKPT_DIR="../cache/packed/anyprec-(${MODEL_PREFIX})-w${BITWIDTH}_orig${BITWIDTH}-redpajama_s1024_blk4096"
    ;;
  lnq)
    if [[ "$SIZE" == "70b" ]]; then
      CKPT_DIR="../cache/layerwise_packed/layerwise-(${MODEL_PREFIX})-w${BITWIDTH}-redpajama_s1024_blk4096_g${GROUPSIZE}_iter2_cd4_nosal"
    else
      CKPT_DIR="../cache/layerwise_packed/layerwise-(${MODEL_PREFIX})-w${BITWIDTH}-redpajama_s1024_blk4096_g${GROUPSIZE}_iter3_cd4_nosal"
    fi
    ;;
  glnq)
    if [[ "$SIZE" == "70b" ]]; then
      CKPT_DIR="../cache/layerwise_packed/layerwise-(${MODEL_PREFIX})-w${BITWIDTH}-redpajama_s1024_blk4096_g${GROUPSIZE}_iter2_cd4"
    else
      CKPT_DIR="../cache/layerwise_packed/layerwise-(${MODEL_PREFIX})-w${BITWIDTH}-redpajama_s1024_blk4096_g${GROUPSIZE}_iter3_cd4"
    fi
    ;;
  *)
    echo "Unsupported method: $METHOD"
    exit 1
    ;;
esac

# Execute the Python script
python sqllm_llama_convert_fuse.py --ckpt_dir "$CKPT_DIR" --bitwidth "$BITWIDTH"
