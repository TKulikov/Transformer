"""tmem — Transformer memory & KV-cache experiments (notebook rewrite).

Package layout:

* :mod:`tmem.memory`   — associative slot memory with TTL (notebook core)
* :mod:`tmem.hooks`    — memory injection into HF models (T5 encoder, decoders)
* :mod:`tmem.kvcache`  — classic vs sparse (sliding/topk) KV-cache policies
* :mod:`tmem.generate` — explicit decoding loop with cache bookkeeping
* :mod:`tmem.dialogue` — strategy x session experiments
* :mod:`tmem.qa`       — pre-scripted questions/answers and probe sessions
* :mod:`tmem.metrics`  — quality + efficiency metrics
* :mod:`tmem.models`   — model registry + tiny offline stand-ins
"""

from .dialogue import DialogueRun, Strategy, default_strategies, run_session
from .generate import DecodeResult, decode_causal, decode_encdec
from .hooks import make_decoder_injection_hook, make_encoder_injection_hook
from .kvcache import ImportanceTracker, KVConfig, keep_indices
from .memory import CorrectionLayer, Memory
from .models import ModelBundle, load_model, tiny_causal, tiny_encdec

__all__ = [
    "CorrectionLayer",
    "DecodeResult",
    "DialogueRun",
    "ImportanceTracker",
    "KVConfig",
    "Memory",
    "ModelBundle",
    "Strategy",
    "decode_causal",
    "decode_encdec",
    "default_strategies",
    "keep_indices",
    "load_model",
    "make_decoder_injection_hook",
    "make_encoder_injection_hook",
    "run_session",
    "tiny_causal",
    "tiny_encdec",
]
