# Originally from https://github.com/IST-DASLab/gptq/blob/main/datautils.py
# Modified to:
# - Only return the test set
# - Skip the tokenization step (return the datasets as-is)

import numpy as np
import torch
from datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2():
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return "\n\n".join(testdata['text'])


def get_ptb():
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return "\n\n".join(valdata['sentence'])


def get_c4(tokenizer, seqlen):
    import random
    valdata = load_dataset(
        "allenai/c4",
        "default",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
    )
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] == seqlen:
            # rare case, discovered with Yi tokenizer
            valenc.append(tmp.input_ids)
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    from transformers.tokenization_utils import BatchEncoding
    valenc = BatchEncoding({'input_ids': valenc, 'attention_mask': torch.ones_like(valenc)})
    return valenc


def get_ptb_new():
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    return " ".join(testdata['sentence'])


def get_ptb_new_sliced():
    raw_text = get_ptb_new()
    sliced = raw_text.replace('<unk>', '< u n k >')
    return sliced


def get_c4_new(tokenizer, seqlen):
    valdata = load_dataset(
        "allenai/c4",
        "default",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
    )
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    from transformers.tokenization_utils import BatchEncoding
    valenc = BatchEncoding({'input_ids': valenc, 'attention_mask': torch.ones_like(valenc)})
    return valenc


def get_loaders(name, tokenizer=None, chunk_size=None):
    if 'wikitext2' in name:
        return get_wikitext2()
    if 'ptb' in name:
        if 'new' in name:
            if 'sliced' in name:
                return get_ptb_new_sliced()
            else:
                return get_ptb_new()
        return get_ptb()
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer, chunk_size)
        return get_c4(tokenizer, chunk_size)

    raise ValueError(f"Unknown dataset {name}")
