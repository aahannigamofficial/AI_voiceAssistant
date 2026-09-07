"""
Speech-to-text transcription using OpenAI Whisper.

This module handles:
- Microphone audio recording
- Converting speech to text
- Error handling for audio input
"""

import pyaudio
import wave
from faster_whisper import WhisperModel
from pathlib import Path
from datetime import datetime


def list_input_devices():
    """Print available input devices and return a list of (index, name) tuples.

    Useful for finding the correct input_device_index to pass to AudioRecorder.record.
    """
    p = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices.append((i, info.get('name')))
                print(f"{i}: {info.get('name')} (inputs={info.get('maxInputChannels')}, rate={info.get('defaultSampleRate')})")
    finally:
        p.terminate()
    return devices


class AudioRecorder:
    """Records audio from microphone to a WAV file."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Audio sample rate (Hz). 16000 is standard for speech.
            chunk_size: Frames per audio buffer.
            channels: 1 for mono, 2 for stereo. Use mono for speech.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        # Use 16-bit PCM for compatibility with WAV/FFmpeg
        self.audio_format = pyaudio.paInt16
        
    def record(self, output_file, duration_seconds=10, silence_threshold=0.01, min_speech_seconds=0.5, input_device_index=None, playback=False):
        """
        Record audio from microphone.
        
        Args:
            output_file: Path to save the .wav file.
            duration_seconds: How long to record (default 10 seconds).
            silence_threshold: Stop early if silence is detected (optional).
            input_device_index: Optional PyAudio device index to use for input. If None, uses default device.
            playback: If True, play back the recording after saving for quick verification.
        
        Returns:
            True if recording successful, False otherwise.
        """
        print(f"\n🎤 Recording for up to {duration_seconds} seconds...")
        print("   (Speak now or press Ctrl+C to stop)")
        
        try:
            p = pyaudio.PyAudio()
            
            # Probe the selected or default input device for its preferred sample rate and channel count
            try:
                if input_device_index is not None:
                    dev_info = p.get_device_info_by_index(int(input_device_index))
                else:
                    dev_info = p.get_default_input_device_info()
                device_rate = int(dev_info.get('defaultSampleRate', self.sample_rate))
                device_channels = int(dev_info.get('maxInputChannels', self.channels))
            except Exception:
                # Fallback to configured values if probing fails
                device_rate = self.sample_rate
                device_channels = self.channels

            # Use device capabilities (fall back to configured values)
            rate = device_rate
            channels = min(self.channels, device_channels) if device_channels > 0 else self.channels

            # Open microphone stream with device-safe parameters
            open_kwargs = dict(
                format=self.audio_format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            if input_device_index is not None:
                open_kwargs['input_device_index'] = int(input_device_index)

            stream = p.open(**open_kwargs)

            frames = []
            silent_frames = 0
            # How many consecutive silent frames count as "silence stop" (2 seconds default)
            max_silent_frames = int(rate / self.chunk_size * 2)
            # How many voiced frames required before allowing early-silence stop
            required_speech_frames = int(max(1, (min_speech_seconds * rate) / self.chunk_size))
            speech_frames_count = 0
            
            # Record audio (use device-selected rate)
            max_rms_norm = 0.0
            for _ in range(0, int(rate / self.chunk_size * duration_seconds)):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Simple silence detection (optional early stop) using RMS
                    import numpy as np
                    # Read as 16-bit PCM to match the stream format
                    audio_data = np.frombuffer(data, dtype=np.int16).astype('float32')
                    # Compute RMS and normalize to 0..1
                    rms = np.sqrt(np.mean(audio_data ** 2))
                    max_int = float(np.iinfo(np.int16).max)
                    rms_norm = rms / max_int if max_int > 0 else 0.0
                    max_rms_norm = max(max_rms_norm, rms_norm)
                    
                    if rms_norm < silence_threshold:
                        silent_frames += 1
                        # Only allow early stop if we've seen enough speech already
                        if silent_frames > max_silent_frames and speech_frames_count >= required_speech_frames:
                            print("   (Silence detected, stopping early)")
                            break
                    else:
                        silent_frames = 0
                        speech_frames_count += 1
                    
                except KeyboardInterrupt:
                    print("   (Recording cancelled)")
                    break
            
            stream.stop_stream()
            stream.close()

            # Retrieve sample width before terminating PyAudio
            sample_width = p.get_sample_size(self.audio_format)
            p.terminate()
            
            # Save to WAV file using the actual stream parameters (rate, channels)
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(output_file), 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
            
            print(f"✓ Saved to {output_file}")

            # Do not send an effectively silent capture to Whisper, which can
            # hallucinate short words such as "You".
            if max_rms_norm < 0.001:
                print(
                    "⚠️ Microphone input is nearly silent "
                    f"(peak RMS={max_rms_norm:.6f}). Check the selected "
                    "microphone and its Windows input level."
                )
                return False

            # Optional playback for verification
            if playback:
                try:
                    print("▶ Playing back recording for verification...")
                    p_play = pyaudio.PyAudio()
                    stream_out = p_play.open(
                        format=self.audio_format,
                        channels=channels,
                        rate=rate,
                        output=True,
                        frames_per_buffer=self.chunk_size
                    )
                    for chunk in frames:
                        stream_out.write(chunk)
                    stream_out.stop_stream()
                    stream_out.close()
                    p_play.terminate()
                except Exception as e:
                    print(f"⚠️ Playback failed: {e}")

            return True
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return False


class SpeechTranscriber:
    """Converts recorded audio to text using Faster-Whisper."""
    
    def __init__(self, model_name="base"):
        """
        Initialize Faster-Whisper transcriber.
        
        Args:
            model_name: Whisper model size.
                - "tiny" (39M) — Fastest, least accurate
                - "base" (140M) — Good balance (default)
                - "small" (244M) — Better accuracy
                - "medium" (769M) — High accuracy, slower
                - "large-v3" (3B) — Highest accuracy, very slow
        """
        print(f"\n📦 Loading Whisper model: {model_name}")
        # device can be "auto", "cuda", or "cpu"
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print("✓ Model loaded")
    
    def transcribe(self, audio_file):
        """
        Convert audio file to text.
        
        Args:
            audio_file: Path to .wav or .mp3 file.
        
        Returns:
            Recognized text (string), or None if transcription failed.
        """
        try:
            print(f"\n🔄 Transcribing {audio_file}...")
            segments, _ = self.model.transcribe(
                str(audio_file),
                language="en",  # Specify English
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )
            # Collect all segments into full text
            text = " ".join([segment.text for segment in segments]).strip()
            print(f"✓ Transcribed: \"{text}\"")
            return text
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None


def record_and_transcribe(duration_seconds=10, model_name="base", input_device_index=None, playback=False, silence_threshold=0.01, min_speech_seconds=0.5):
    """
    Simple helper: record audio and transcribe in one call.
    
    Args:
        duration_seconds: How long to record.
        model_name: Whisper model size.
    
    Returns:
        Recognized text (string), or None if failed.
    """
    # Create temp audio file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = Path("./temp") / f"audio_{timestamp}.wav"
    
    # Record
    recorder = AudioRecorder()
    if not recorder.record(audio_file, duration_seconds=duration_seconds, silence_threshold=silence_threshold, min_speech_seconds=min_speech_seconds, input_device_index=input_device_index, playback=playback):
        return None
    
    # Transcribe
    transcriber = SpeechTranscriber(model_name=model_name)
    text = transcriber.transcribe(audio_file)
    
    return text