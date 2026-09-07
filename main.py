"""
Phase 1, Step 1: Speech-to-Text Demo

This script:
1. Records audio from your microphone for 10 seconds
2. Converts it to text using Whisper
3. Prints the result

Run with: python main.py
"""

from app.speech import record_and_transcribe


def main():
    print("\n" + "="*60)
    print("PHASE 1, STEP 1: SPEECH-TO-TEXT")
    print("="*60)
    print("\nThis script will:")
    print("1. Record audio from your microphone")
    print("2. Convert speech to text using Whisper")
    print("3. Print the recognized text")
    print("\n" + "-"*60)
    
    # Record and transcribe
    text = record_and_transcribe(duration_seconds=10, model_name="base")
    
    if text:
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"You said: \"{text}\"")
        print("="*60)
    else:
        print("\n❌ Failed to transcribe audio.")


if __name__ == "__main__":
    main()