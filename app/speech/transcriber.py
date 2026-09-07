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
        self.audio_format = pyaudio.paFloat32
        
    def record(self, output_file, duration_seconds=10, silence_threshold=0.02):
        """
        Record audio from microphone.
        
        Args:
            output_file: Path to save the .wav file.
            duration_seconds: How long to record (default 10 seconds).
            silence_threshold: Stop early if silence is detected (optional).
        
        Returns:
            True if recording successful, False otherwise.
        """
        print(f"\n🎤 Recording for up to {duration_seconds} seconds...")
        print("   (Speak now or press Ctrl+C to stop)")
        
        try:
            p = pyaudio.PyAudio()
            
            # Open microphone stream
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            silent_frames = 0
            max_silent_frames = int(self.sample_rate / self.chunk_size * 2)  # 2 seconds of silence
            
            # Record audio
            for i in range(0, int(self.sample_rate / self.chunk_size * duration_seconds)):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Simple silence detection (optional early stop)
                    import numpy as np
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    volume = np.abs(audio_data).mean()
                    
                    if volume < silence_threshold:
                        silent_frames += 1
                        if silent_frames > max_silent_frames and len(frames) > 10:
                            print("   (Silence detected, stopping early)")
                            break
                    else:
                        silent_frames = 0
                    
                except KeyboardInterrupt:
                    print("   (Recording cancelled)")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save to WAV file
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(output_file), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            print(f"✓ Saved to {output_file}")
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
            segments, info = self.model.transcribe(
                str(audio_file),
                language="en",  # Specify English
                beam_size=5
            )
            # Collect all segments into full text
            text = " ".join([segment.text for segment in segments]).strip()
            print(f"✓ Transcribed: \"{text}\"")
            return text
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None


def record_and_transcribe(duration_seconds=10, model_name="base"):
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
    if not recorder.record(audio_file, duration_seconds=duration_seconds):
        return None
    
    # Transcribe
    transcriber = SpeechTranscriber(model_name=model_name)
    text = transcriber.transcribe(audio_file)
    
    return text