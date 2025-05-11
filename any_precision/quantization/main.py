
import os
import os.path
import shutil
import logging

from .config import *
from ..analyzer import get_analyzer
from .quantize import seed_and_upscale
from .pack import pack
from .datautils import get_tokens
from .gradients import get_gradients
import datetime
import sys

# Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def any_precision_quantize(
        model,
        seed_precision=DEFAULT_SEED_PRECISION,
        parent_precision=DEFAULT_PARENT_PRECISION,
        mode='pack',
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,
        dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
        cpu_count=None,
        overwrite_tokens=False,
        overwrite_gradients=False,
        overwrite_quantize=False,
        overwrite_pack=False,
        random_state=None,
        dns=False,
        num_groups=None,
        sub_saliency=None,
        skip_save_gradients=False,
):

    # Logging with time sans date, level name, and message
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/quantize_log_{timestamp}.txt"),
        ]
    )

    assert mode in ['tokens', 'gradients', 'quantize', 'pack'], \
        "mode must be one of 'tokens', 'gradients', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."

    if overwrite_tokens:
        if not overwrite_gradients:
            logging.warning("Gradients need to be recalculated if tokens are recalculated. "
                            "Setting overwrite_gradients to True.")
            overwrite_gradients = True

    if overwrite_gradients:
        if not overwrite_quantize:
            logging.warning("Parent model needs to be recalculated if gradients are recalculated. "
                            "Setting overwrite_quantize to True.")
            overwrite_quantize = True

    if overwrite_quantize:
        if not overwrite_pack:
            logging.warning("Packed model needs to be recalculated if parent model is recalculated. "
                            "Setting overwrite_pack to True.")
            overwrite_pack = True

    if mode == 'tokens':
        logging.info("Running: [Tokens]")
    elif mode == 'gradients':
        logging.info("Running: [Tokens -> Gradients]")
    elif mode == 'quantize':
        logging.info("Running: [Tokens -> Gradients -> Quantize]")
    else:
        logging.info("Running: [Tokens -> Gradients -> Quantize -> Pack]")

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    logging.info(f"Running Any-Precision Quantization on {model_name} with seed precision {seed_precision} and "
                 f"parent precision {parent_precision} using {dataset} for gradient calculation")

    # ------------------- Load model -------------------

    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)

    # ------------------- Set cache paths -------------------

    tokens_cache_path = (f"{cache_dir}/tokens/"
                         f"({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt")

    gradients_cache_path = (f"{cache_dir}/gradients/"
                            f"({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt")
    
    if num_groups is not None:
        saliency_cache_path = (f"{cache_dir}/saliency/"
                            f"({model_name})-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}")
    else:
        saliency_cache_path = None

    quantized_cache_path = (f"{cache_dir}/quantized/"
                          f"{'dns-' if dns else ''}({model_name})-w{parent_precision}_orig{seed_precision}"
                          f"-{dataset}_s{num_examples}_blk{seq_len}")

    model_output_path = (f"{cache_dir}/packed/"
                         f"anyprec-({model_name})-w{parent_precision}_orig{seed_precision}"
                         f"-{dataset}_s{num_examples}_blk{seq_len}")

    logging.info(f"Tokens cache path: {tokens_cache_path}")
    logging.info(f"Gradients cache path: {gradients_cache_path}")
    logging.info(f"Saliency cache path: {saliency_cache_path}")
    logging.info(f"Quantized cache path: {quantized_cache_path}")
    logging.info(f"Model output path: {model_output_path}")

    # ------------------- Get tokens -------------------

    logging.info("------------------- Get tokens -------------------")
    logging.info(f"Getting tokens for {dataset} with sequence length {seq_len} and {num_examples} examples")
    tokens = get_tokens(dataset, "train", analyzer.tokenizer, seq_len, num_examples, tokens_cache_path, random_state)
    logging.info("Tokens loading complete.")

    if mode == 'tokens':
        return

    # ------------------- Gradients -------------------

    logging.info("------------------- Gradients -------------------")

    logging.info("Beginning gradient calculation...")
    # Calculate or load gradients
    if overwrite_gradients and os.path.exists(gradients_cache_path):
        # if the user wants to recalculate the gradients, delete the cached gradients
        logging.info(f"Detected cached gradients at {gradients_cache_path}. Will delete and recalculate.")
        os.remove(gradients_cache_path)

    model_gradients = get_gradients(
        analyzer=analyzer,
        input_tokens=tokens,
        save_path=gradients_cache_path,
        saliency_path=saliency_cache_path,
        num_groups=num_groups,
        sub_saliency=sub_saliency,
        skip_save_gradients=skip_save_gradients,
    )
    
    logging.info("Gradient calculation complete.")

    if mode == 'gradients':
        return

    # ------------------- Quantize: Seed + Upscale -------------------

    logging.info("------------------- Quantize: Seed + Upscale -------------------")

    # Calculate or load parent
    logging.info(f"Beginning {seed_precision}~{parent_precision}-bit Any-Precision Quantization...")
    # Note that this saves the seed model to the cache path and must be loaded for the upscale step
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        # if the user wants to recalculate the seed, delete the cached seed
        logging.info(f"Detected cached parent at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)

    # this skips over existing layers in the cache, and doesn't overwrite them
    seed_and_upscale(
        analyzer=analyzer,
        gradients=model_gradients,
        output_folder=quantized_cache_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        random_state=random_state,
    )

    if mode == 'quantize':
        return

    del model_gradients  # free up memory
    analyzer.drop_original_weights()  # drop the original weights to save memory

    logging.info("Quantization(Seed + Upscale) complete.")

    # ------------------- Pack -------------------
    logging.info("------------------- Pack -------------------")

    # check for non-empty directory
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        if overwrite_pack:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will delete and "
                         f"re-pack.")
            shutil.rmtree(model_output_path)
        else:
            # if the user doesn't want to overwrite the pack, but the directory is not empty, skip packing
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will skip packing.")
            return

    pack(
        analyzer=analyzer,
        lut_path=quantized_cache_path,
        output_model_path=model_output_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        dns=dns,
    )

    logging.info("Packing complete.")
