"""
Offline wake-word detection using openWakeWord.

The detector runs locally and does not require an account or access key.
"""

import time
import os
from pathlib import Path

import numpy as np
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import download_models


DEFAULT_WAKE_WORDS = (
    "alexa",
    "hey_jarvis",
    "hey_mycroft",
    "hey_rhasspy",
    "timer",
    "weather",
)
WAKE_WORD_ALIASES = {}


def list_wake_words():
    """Return the built-in openWakeWord model names."""
    return DEFAULT_WAKE_WORDS


class WakeWordDetector:
    """Detect a built-in or custom openWakeWord model from microphone audio."""

    sample_rate = 16_000
    frame_length = 1_280

    def __init__(
        self,
        wake_word="hey_mycroft",
        sensitivity=0.5,
        input_device_index=None,
        keyword_path=None,
    ):
        """
        Args:
            wake_word: Built-in model name, or the name used by a custom model.
            sensitivity: Detection threshold from 0.0 to 1.0. Lower values
                         are more sensitive and may produce more false positives.
            input_device_index: PyAudio device index. If None, uses the default.
            keyword_path: Path to a custom openWakeWord model (.tflite/.onnx).
        """
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.input_device_index = input_device_index
        configured_path = keyword_path
        self.keyword_path = Path(configured_path) if configured_path else None
        self.is_running = False
        self.model_key = WAKE_WORD_ALIASES.get(self.wake_word, self.wake_word)
        self.model = self._init_model()

    def _init_model(self):
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError("sensitivity must be between 0.0 and 1.0.")

        model_paths = [str(self.keyword_path)] if self.keyword_path else None
        if self.model_key not in DEFAULT_WAKE_WORDS and not self.keyword_path:
            raise ValueError(
                f'"{self.wake_word}" is not built in. Provide a custom openWakeWord '
                "model with keyword_path."
            )
        try:
            if model_paths:
                model = Model(wakeword_models=model_paths, inference_framework="onnx")
            else:
                download_models(model_names=[self.model_key])
                model = Model(
                    wakeword_models=[self.model_key],
                    inference_framework="onnx",
                )
        except Exception as error:
            raise RuntimeError(
                "Could not load the openWakeWord model. Check your internet "
                "connection for the first model download, then try again."
            ) from error

        available = set(model.models)
        if self.keyword_path:
            if len(available) != 1:
                raise ValueError(
                    "A custom openWakeWord model must contain exactly one model."
                )
            self.model_key = next(iter(available))
        elif self.model_key not in available:
            model = None
            raise ValueError(
                f'Wake word "{self.wake_word}" is unavailable. '
                f"Choose one of: {', '.join(sorted(available))}."
            )
        print(f'Wake-word detector initialized for: "{self.wake_word}"')
        return model

    def listen_for_wake_word(self, timeout_seconds=None, on_wake_word_detected=None):
        """Listen until the wake word is detected, stopped, or timed out."""
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        print(f'\nListening for wake word: "{self.wake_word}"')
        print("(Press Ctrl+C to stop listening)")
        self.is_running = True
        audio = pyaudio.PyAudio()
        stream = None
        started_at = time.monotonic()

        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_length,
                input_device_index=self.input_device_index,
            )
            while self.is_running:
                pcm = stream.read(self.frame_length, exception_on_overflow=False)
                scores = self.model.predict(np.frombuffer(pcm, dtype=np.int16))
                if scores.get(self.model_key, 0.0) >= self.sensitivity:
                    print(f'\nWake word detected: "{self.wake_word}"')
                    if on_wake_word_detected:
                        on_wake_word_detected(0)
                    return True
                if (
                    timeout_seconds is not None
                    and time.monotonic() - started_at >= timeout_seconds
                ):
                    print("\nTimeout reached")
                    break
        except KeyboardInterrupt:
            print("\nListening stopped")
        except Exception as error:
            print(f"Error during wake-word detection: {error}")
            return False
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()
            self.is_running = False
        return False

    def stop(self):
        """Request that an active listening loop stop."""
        self.is_running = False
