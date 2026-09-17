"""OpenVoiceOS VAD plugin backed by vadonnx.

One plugin exposes every model that ``vadonnx.list_models()`` reports. The ``model``
config key selects it.
"""
from typing import Optional

import numpy as np
from ovos_plugin_manager.templates.vad import VADEngine
from ovos_utils.log import LOG
from vadonnx import list_models, load_vad

DEFAULT_MODEL = "silero"


class VadonnxVAD(VADEngine):
    """``VADEngine`` that runs a vadonnx model.

    Config keys:
        model: vadonnx model name, local ``.onnx`` path or URL (default ``silero``).
        threshold: speech probability at or above which a chunk is speech. When
            omitted, the default of the vadonnx backend applies.
        revision: HuggingFace revision override for the model download.
        cache_dir: download cache directory override.
        providers: ONNX Runtime execution providers.
        signature: IO signature, required for a custom ``.onnx`` file without a
            sidecar signature.
    """

    def __init__(self, config: Optional[dict] = None, sample_rate: Optional[int] = None):
        super().__init__(config, sample_rate)
        self.model_name = self.config.get("model") or DEFAULT_MODEL
        kwargs = {}
        for key in ("threshold", "revision", "cache_dir", "providers", "signature"):
            if self.config.get(key) is not None:
                kwargs[key] = self.config[key]
        self.vad = load_vad(self.model_name, **kwargs)
        self.threshold = self.vad.threshold
        LOG.info(f"vadonnx VAD loaded model={self.model_name} "
                 f"threshold={self.threshold} frame={self.vad.frame_duration * 1000:.0f}ms")

    @staticmethod
    def available_models():
        return list_models()

    def reset(self):
        self.vad.reset()

    def is_silence(self, chunk: bytes) -> bool:
        audio = np.frombuffer(chunk, dtype=np.int16)
        prob = self.vad.process_chunk(audio, sample_rate=self.sample_rate)
        return prob < self.threshold

    def extract_speech(self, audio: bytes) -> Optional[bytes]:
        # the model keeps streaming state and a partial frame between calls;
        # a full-buffer pass must start and end clean
        self.vad.reset()
        try:
            return super().extract_speech(audio)
        finally:
            self.vad.reset()
