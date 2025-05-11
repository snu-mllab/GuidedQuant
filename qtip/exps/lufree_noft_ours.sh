# get argument

# Example script to quantize Llama 2 7b to 2 bits
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 [METHOD] [MODEL_SIZE] [BIT]"
  echo "Example: $0 1mad 7b 2"
  exit 1
fi

# Fill these in with your own paths
METHOD=$1 # one of 1mad, 3inst, hyb
MODEL_SIZE=$2 # one of 7b, 13b, 70b
BIT=$3 # one of 1, 2, 3, 4

CKPT=./$METHOD/models
HF=./$METHOD/hf
LOG=./$METHOD/logs
EXPNAMENOFT=2_${MODEL_SIZE}_${BIT}_saliency_no_ft

if [ $MODEL_SIZE = "7b" ]; then
  HESS="../cache/hessians/(Llama-2-7b-hf)-redpajama_s1024_blk4096_g4"
  GSAL=4
elif [ $MODEL_SIZE = "13b" ]; then
  HESS="../cache/hessians/(Llama-2-13b-hf)-redpajama_s1024_blk4096_g4"
  GSAL=4
elif [ $MODEL_SIZE = "70b" ]; then
  HESS="../cache/hessians/(Llama-2-70b-hf)-redpajama_s1024_blk4096_g2"
  GSAL=2
elif [ $MODEL_SIZE = "70b_g1" ]; then
  HESS="../cache/hessians/(Llama-2-70b-hf)-redpajama_s1024_blk4096_g1"
  GSAL=1
  MODEL_SIZE="70b"
else
  echo "MODEL_SIZE must be one of 7b, 13b, 70b"
  exit 1
fi

if [ $METHOD = "1mad" ]; then
  VECSIZE=1
  TLUTBITS=0
  MANIFEST="--manifest"
  RECONS="--ft_grad_ckpt" # just placeholder
elif [ $METHOD = "3inst" ]; then
  VECSIZE=1
  TLUTBITS=0
  MANIFEST="--manifest"
  RECONS="--ft_grad_ckpt" # just placeholder
elif [ $METHOD = "hyb" ]; then
  VECSIZE=2
  TLUTBITS=9
  METHOD="quantlut_sym"
  MANIFEST="--manifest" # since hyb has kernel, one doesn't need to use --manifest except for bit=1
  RECONS="--ft_train_recons"
else
  echo "METHOD must be one of 1mad, 3inst, hyb"
  exit 1
fi

echo "METHOD: $METHOD"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "BIT: $BIT"
echo "VECSIZE: $VECSIZE"
echo "TLUTBITS: $TLUTBITS"
echo "EXPNAMENOFT: $EXPNAMENOFT"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

# main quantization script
python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/$EXPNAMENOFT \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-${MODEL_SIZE}-hf \
       --in_hess_path $HESS \
       --scale_override 0.9 \
       --ft_epochs 0 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K ${BIT} \
       --V ${VECSIZE} \
       --decode_mode ${METHOD} \
       --tlut_bits ${TLUTBITS} \
       --use_saliency \
       --g_sal ${GSAL} \
       --ft_grad_ckpt \
       --devset_size 2 \
       --ft_valid_size 1 ${RECONS}\
       2>&1 | tee -a $LOG/$EXPNAMENOFT

# convert the quantized model to a hf model
python -m quantize_llama.hfize_llama --quantized_path $CKPT/$EXPNAMENOFT --hf_output_path $HF/$EXPNAMENOFT >> $LOG/$EXPNAMENOFT 2>&1 

# evaluate perplexity and zeroshot results
python -m eval.eval_ppl  --hf_path $HF/$EXPNAMENOFT $MANIFEST >> $LOG/$EXPNAMENOFT 2>&1
# python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/$EXPNAMENOFT >> $LOG/$EXPNAMENOFT 2>&1
