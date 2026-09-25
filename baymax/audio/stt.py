"""Local speech-to-text.

The browser segments speech with an energy VAD and posts 16 kHz mono PCM WAV
segments to /api/stt. This module transcribes them locally with
faster-whisper when it is installed; otherwise the endpoint reports that
local STT is unavailable and the UI offers browser speech recognition (which
requires the `cloud_speech_recognition` consent, since some browsers send
audio to their vendor).
"""
from __future__ import annotations

import io
import math
import os
import threading
import wave
from dataclasses import dataclass
from typing import Optional


@dataclass
class STTResult:
    text: str
    language: Optional[str]
    confidence: Optional[float]
    model_id: str
    no_speech_prob: Optional[float] = None


class LocalSTT:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self.size = os.environ.get("BAYMAX_WHISPER_MODEL", "small")
        self.error: Optional[str] = None
        try:
            import faster_whisper  # noqa: F401
            self.available = os.environ.get("BAYMAX_STT", "local") != "off"
        except Exception as e:
            self.available = False
            self.error = f"faster-whisper not installed ({type(e).__name__})"

    @property
    def model_id(self) -> str:
        return f"faster_whisper:{self.size}"

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.size, device=os.environ.get("BAYMAX_WHISPER_DEVICE", "auto"),
                                       compute_type=os.environ.get("BAYMAX_WHISPER_COMPUTE", "int8"))
        return self._model

    def transcribe(self, wav_bytes: bytes, language_hint: Optional[str] = None) -> STTResult:
        import numpy as np

        with wave.open(io.BytesIO(wav_bytes)) as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError("expected 16-bit mono WAV")
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if rate != 16000:
            idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / rate))
            pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
        hint = None
        if language_hint and language_hint != "auto":
            hint = language_hint.split("-")[0]
        with self._lock:
            model = self._load()
            segments, info = model.transcribe(pcm, language=hint, beam_size=1, vad_filter=True,
                                              condition_on_previous_text=False)
            segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs:
            return STTResult("", info.language, 0.0, self.model_id, 1.0)
        avg_lp = sum(s.avg_logprob for s in segs) / len(segs)
        nsp = max(s.no_speech_prob for s in segs)
        conf = max(0.0, min(1.0, math.exp(avg_lp)))
        if nsp > 0.6 and conf < 0.5:  # classic Whisper hallucination on noise
            text = ""
        return STTResult(text, info.language, round(conf, 3), self.model_id, round(nsp, 3))
