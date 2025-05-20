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
</p>

# News
- **May, 2025**: GuidedQuant is accepted to **ICML 2025**.

# Overview
![Light Mode](assets/objective-light.png#gh-light-mode-only)
![Dark Mode](assets/objective-dark.png#gh-dark-mode-only)

> *<b>GuidedQuant</b> enhances LLM quantization by integrating gradient information from the end loss into the quantization objective, boosting the performance of SOTA weight-only scalar, weight-only vector, and weight-and-activation quantization. Additionally, we introduce <b>LNQ</b>, a non-uniform scalar quantization algorithm which is guaranteed to monotonically decrease the quantization objective value.*

# Installation 

1. Install the requirements (we used Python 3.11 version).
      ```bash
      pip install -r requirements.txt
      ```
2. Install the Any-Precision CUDA kernels.
      ```bash
      cd anyprecision/modules/kernels
      pip install .
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
We upload the pre-quantized models for Llama-2 and Llama-3 models in the following Hugging Face collections.

| Method | Link |
|:---|:---:|
| SqueezeLLM        | **[Link](https://huggingface.co/collections/jusjinuk/guidedquant-squeezellm-682ca2b6d71351d9bd94e94d)** |
| LNQ               | **[Link](https://huggingface.co/collections/jusjinuk/guidedquant-lnq-682c879c799d0ba767b57216)** |
| LNQ + GuidedQuant | **[Link](https://huggingface.co/collections/jusjinuk/guidedquant-lnq-gquant-682c89b60907f4a88caf6fa3)** |

You could easily load and test them (e.g., using the following code).

```python
from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
from transformers import AutoTokenizer, TextStreamer

# method = LNQ + GuidedQuant / bits = 3 / num_groups = 1
quantized_model_name = "jusjinuk/layerwise-Meta-Llama-3-8B-w3-redpajama_s1024_blk4096_g1_iter3_cd4"
model = AnyPrecisionForCausalLM.from_quantized(quantized_model_name)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_name)
streamer = TextStreamer(tokenizer)

prompt = "Harry, Ron, and Hermione are"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

model.generate(inputs["input_ids"], 
    max_length=200, do_sample=True, temperature=1.0, streamer=streamer
)
```

# Usage

### Download the calibration data
We provide the tokenized calibration data for Llama-2 and Llama-3 to reproduce the results in the paper.
```bash
bash scripts/download_calibration.sh
```


### Weight-only scalar quantization (SqueezeLLM, LNQ + GuidedQuant)
Below command saves the gradient and quantizes model with SqueezeLLM (e.g., `bash scripts/run_sqllm.sh meta-llama/Llama-2-7b-hf 2 4`).
```bash
bash scripts/run_sqllm.sh $MODEL_NAME $BITS $NUM_GROUPS
```
<!-- **Note**:  -->

For LNQ + GuidedQuant, run the following command (e.g., `bash scripts/run_lnq.sh meta-llama/Llama-2-7b-hf 2 4`).
```bash
bash scripts/run_lnq.sh $MODEL_NAME $BITS $NUM_GROUPS
```
<!-- **Note**:  -->


### Weight-only vector quantization (QTIP + GuidedQuant)

To be updated.



### Evaluation

Run the following command to evaluate the performance of the pre-trained / quantized / pre-quantized models.
```python
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


