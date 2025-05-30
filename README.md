<p align=center>
<div align=center>
<img src="assets/guidedquant-logo.png" width=350>
</div>
<h1 align="center">GuidedQuant</h1>
</p>
<p align="center"><b>Smarter LLM Post-Training Quantization using End Loss Guidance</b>, boosting the performance of <br> state-of-the-art <i>weight-only scalar</i>, <i>weight-only vector</i>, and <i>weight-and-activation</i> quantization methods.</p>
<p align="center">
<a href="https://arxiv.org/abs/2505.07004"><img src="https://img.shields.io/badge/arXiv-2505.07004-b31b1b.svg"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow"></a>
<a href="https://jusjinuk.me/blog/guidedquant/"><img src="https://img.shields.io/badge/Blog-GuidedQuant-blue"></a>
</p>

# News
- **May, 2025**: GuidedQuant is accepted to **ICML 2025**.

# Overview
![Light Mode](assets/objective-light.png#gh-light-mode-only)
![Dark Mode](assets/objective-dark.png#gh-dark-mode-only)

> *<b>GuidedQuant</b> enhances LLM quantization by integrating gradient information from the end loss into the quantization objective, boosting the performance of SOTA weight-only scalar, weight-only vector, and weight-and-activation quantization. Additionally, we introduce <b>LNQ</b>, a non-uniform scalar quantization algorithm which is guaranteed to monotonically decrease the quantization objective value.*

# Installation 

1. Install the requirements (we used Python 3.11 / CUDA 12.4 version).
      ```bash
      pip install -r requirements.txt
      ```
2. Install the Any-Precision-LLM CUDA kernels.

      Install either from source (this might take a while),
      ```bash
      cd inference/ap_gemv
      bash install.sh
      ```
      or from pre-built binaries,
      ```bash
      # CUDA 12.4
      pip install ap-gemv -i https://jusjinuk.me/whl/cu124
      ```
3. (Optional) To reproduce weight-only vector & weight-and-activation quantization results, install the following dependencies.
      ```bash
      git clone https://github.com/Dao-AILab/fast-hadamard-transform 
      cd fast-hadamard-transform
      pip install .
      cd ..
      cd qtip/qtip-kernels
      pip install .
      ```


# Pre-quantized Models

### Pre-trained models.

| Type | Models | Method | Link |
|:---|:---|:---|:---:|
|Pre-trained models | `Llama-2-7b-hf`, `Llama-2-13b-hf`, `Llama-2-70b-hf`, `Meta-Llama-3-8B`, `Meta-Llama-3-70B` | SqueezeLLM        | **[Link](https://huggingface.co/collections/jusjinuk/guidedquant-squeezellm-682ca2b6d71351d9bd94e94d)** |
|      | | LNQ               | **[Link](https://huggingface.co/collections/jusjinuk/guidedquant-lnq-682c879c799d0ba767b57216)** |
|      | | LNQ + GuidedQuant | **[Link](https://huggingface.co/collections/jusjinuk/guidedquant-lnq-gquant-682c89b60907f4a88caf6fa3)** |

### Instruction-tuned models.

| Type | Models | Link |
|:---|:---|:---:|
| Instruction-tuned models        | `Llama-3.1-8B-Instruct`, `Llama-3.3-70B-Instruct` | **[Link](https://huggingface.co/collections/jusjinuk/instruction-tuned-models-guidedquant-68334269c44cd3eb21f7bd61)** |

### Demo

You could easily load and test them using `AnyPrecisionForCausalLM` class, as shown in the following example (runs on one RTX 3090).

```python
from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
from transformers import AutoTokenizer, TextStreamer

# model: Llama-3.3-70B-Instruct / method = GuidedQuant + LNQ / bits = 2 / num_groups = 1
quantized_model_name = "jusjinuk/Llama-3.3-70B-Instruct-2bit-GuidedQuant-LNQ"
model = AnyPrecisionForCausalLM.from_quantized(quantized_model_name)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_name)
streamer = TextStreamer(tokenizer)

prompt = "Write me a short and concise story about Harry, Ron, and Hermione.\n"
chat = [
    {"role": "system", "content": "You are a helpful assistant.\n"},
    {"role": "user", "content": prompt},
]

inputs = tokenizer.apply_chat_template(
    chat, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(model.device)

model.generate(inputs, 
    max_new_tokens=200, do_sample=False, temperature=1.0, streamer=streamer, pad_token_id=tokenizer.eos_token_id
)
```

# Inference Speed-up

We provide a simple [inference script (~80 LOC)](./inference_example.py) that uses `torch.compile` with Hugging Face `generate` function, showing the speed-up of LNQ + GuidedQuant quantized model, using Any-Precision-LLM kernel (`ap-gemv` kernel). This example is inspired by the demo code of [Any-Precision-LLM](https://github.com/SNU-ARC/any-precision-llm).
```bash
# pre-trained Llama-3.1-8B-Instruct
# ~43 tok/s in one RTX 3090
python inference_example.py

# LNQ + GuidedQuant quantized Llama-3.1-8B-Instruct (bits=2)
# ~130 tok/s in one RTX 3090
python inference_example.py -q
```

In the paper, we report the further-optimized throughput of each model obtained by fusing the Q/K/V layer and the Up/Gate layer within every Transformer block.

<details>
<summary><b>Click to expand the commands for reproducing the throughput results in the paper.</b></summary>
First, do `cd inference/`, and then

1. For quantized models, run
      ```bash
      python sqllm_llama_convert_fuse.py --ckpt_dir <path_to_quantized_ckpt> --bitwidth <bitwidth>
      python generate.py --compile 2 --num_samples 5 \
            --model_name ${model} --bitwidth ${BITWIDTH} --dtype "float16" \
            --checkpoint_path ${checkpoint_path} \
            --backend ap --max_new_tokens 100
      ```

2. For pre-trained models (without quantization), run
      ```bash
      python pt_llama_convert_fuse.py --ckpt_dir <save_path> --model_name <huggingface_model_name>
      python generate.py --compile 2 --num_samples 5 \
            --model_name ${model} --bitwidth 16 --dtype "float16" \
            --checkpoint_path ${checkpoint_path} \
            --max_new_tokens 100
      ```
</details>

# Usage

### Download the calibration data
We provide the tokenized calibration data for Llama-2 and Llama-3 to reproduce the results in the paper.
```bash
bash scripts/download_calibration.sh
```


### Weight-only scalar quantization (SqueezeLLM, LNQ + GuidedQuant)
Below command saves the weight gradients and activation gradients (averaged into $NUM_GROUPS groups) and quantizes model with SqueezeLLM.
```bash
bash scripts/run_sqllm.sh $MODEL_NAME $BITS $NUM_GROUPS
# e.g., bash scripts/run_sqllm.sh meta-llama/Llama-2-7b-hf 2 4
```
<!-- **Note**:  -->

Afterwards, for LNQ + GuidedQuant, run the following command.
```bash
bash scripts/run_lnq.sh $MODEL_NAME $BITS $NUM_GROUPS
# e.g., bash scripts/run_lnq.sh meta-llama/Llama-2-7b-hf 2 4
```
<!-- **Note**:  -->


### Weight-only vector (QTIP + GuidedQuant)

To be updated.

### Weight-and-activation quantization (SpinQuant + GuidedQuant)

To be updated.



### Evaluation

Run the following command to evaluate the performance of the pre-trained / quantized / pre-quantized models.
```bash
python run_eval.py
```

Add `--downstream` option to evaluate on downstream tasks using `lm-eval-harness` library.




## Acknowledgement
This code heavily relies on the following repositories:
- [Any-Precision-LLM](https://github.com/SNU-ARC/any-precision-llm)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [SpinQuant](https://github.com/facebookresearch/SpinQuant)
- [Fast Hadamard Transform](https://github.com/Dao-AILab/fast-hadamard-transform)

We thank the authors for their open-source implementations and contributions to the community.

## Citation

Please cite our paper if you find our work useful:

```
@inproceedings{kim2025guidedquant,
      title={GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance}, 
      author={Jinuk Kim and Marwa El Halabi and Wonpyo Park and Clemens JS Schaefer and Deokjae Lee and Yeonhong Park and Jae W. Lee and Hyun Oh Song},
      booktitle = {International Conference on Machine Learning (ICML)},
      year={2025},
}
```


