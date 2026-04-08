# ---------------------------
# 1) Text vocabulary / tasks
# ---------------------------

import numpy as np

PAD_TOKEN = "<pad>"
VOCAB = {
    PAD_TOKEN: 0,
    "run": 1,
    "forward": 2,
    "backward": 3,
    "slowly": 4,
}
ID2TOKEN = {v: k for k, v in VOCAB.items()}

TASKS = {
    0: "run forward",
    1: "run backward",
    2: "run slowly",
}

MAX_TEXT_LEN = 3
IMAGE_SIZE = (36, 36)  # H, W

def tokenize(text: str, max_len: int = MAX_TEXT_LEN) -> np.ndarray:
    ids = [VOCAB.get(tok, 0) for tok in text.lower().split()]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [VOCAB[PAD_TOKEN]] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)
