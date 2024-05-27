import torch

from typing import Optional
from dataclasses import dataclass

from tokenizer import SimpleTokenizer


@dataclass
class ModelArgs:
    dim: int = 128  # 4096
    n_layers: int = 12  # 32
    n_heads: int = 4  # 32
    n_kv_heads: Optional[int] = 2  # None
    vocab_size: int = 512  # -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000  # 500000

    max_batch_size: int = 24  # 32
    max_seq_len: int = 512  # 8192 but their maximum chunk size when running inference is 2048

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate: float = 0.1

    tokenizer: SimpleTokenizer = SimpleTokenizer.get_tokenizer(512)
