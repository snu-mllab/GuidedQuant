from datasets import load_dataset
import random
import numpy as np
import logging
import torch
from tqdm import tqdm
import os

def _get_wikitext2(split):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"

    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
    return data['text']


def _get_ptb(split, slice_unk=True):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for ptb"

    data = load_dataset('ptb_text_only', 'penn_treebank', split=split,
                        trust_remote_code=True)
    data_list = data['sentence']

    if slice_unk:
        data_list = [s.replace('<unk>', '< u n k >') for s in data_list]

    return data_list


def _get_c4(split):
    assert split in ['train', 'validation'], f"Unknown split {split} for c4"

    if split == 'train':
        data = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
            trust_remote_code=True
        )
    else:
        assert split == 'validation'
        data = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
            trust_remote_code=True
        )

    return data['text']


def _get_pileval(split):
    if split != 'validation':
        logging.warning(f"Pileval only has a validation split, but got split={split}. Using validation split.")
    data = load_dataset("mit-han-lab/pile-val-backup", split="validation", trust_remote_code=True)

    return data['text']


def _get_redpajama(split):
    assert split in ['train'], "RedPajama only has a train split"
    data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split, trust_remote_code=True)
    return data['text']


def _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        tokens = tokens[:seq_len]

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples

def _sample_and_tokenize_from_middle(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()
    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        seq_start = random.randint(0, len(tokens) - seq_len)

        tokens = tokens[seq_start:seq_start + seq_len]
        assert tokens.shape[-1] == seq_len, f"Token length {len(tokens)} != seq_len {seq_len}"

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()
    return samples


def _sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
    f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    logging.info(f"Tokenizing {len(texts)} texts")
    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
    samples = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {num_samples} samples of length {seq_len}")
    while len(samples) < num_samples:
        idx = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        
        if selected_indices:
            closest_idx = min(selected_indices, key=lambda x: abs(x - idx), default=idx)
            if idx <= closest_idx + seq_len and idx >= closest_idx - seq_len:
                continue

        j = idx + seq_len
        inp = trainenc.input_ids[:, idx:j]
        tokens = inp.clone()
        tokens = tokens.squeeze(0)

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples


def _get_dataset(dataset_name, split):
    if dataset_name == 'wikitext2':
        return _get_wikitext2(split)
    elif dataset_name == 'ptb':
        return _get_ptb(split)
    elif dataset_name == 'c4':
        return _get_c4(split)
    elif dataset_name == 'pileval':
        return _get_pileval(split)
    elif dataset_name == 'redpajama':
        return _get_redpajama(split)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(dataset_name, split, tokenizer, seq_len, num_samples, save_path=None, seed=None):

    if save_path is not None and os.path.isfile(save_path):
        logging.info(f"Loading tokens from {save_path}")
        return torch.load(save_path)

    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(dataset_name, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    if dataset_name == 'wikitext2':
        tokens = _sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)
    elif dataset_name == 'redpajama':
        # Following PV-Tuning Github.
        tokens = _sample_and_tokenize_from_middle(texts, tokenizer, seq_len, num_samples, seed)
    else:
        tokens = _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)

    if save_path is not None:
        logging.info(f"Saving tokens to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tokens, save_path)

    return tokens
