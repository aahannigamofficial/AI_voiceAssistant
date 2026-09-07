"""
Phase 1, Step 2: Wake-Word Detection Demo

This script:
1. Listens continuously for "Hey Mycroft"
2. When detected, records the command
3. Transcribes the command
4. Prints the result

Run with: python test_wake_word.py
"""

from app.wake_word import WakeWordDetector
from app.speech import record_and_transcribe


def main():
    print("\n" + "="*60)
    print("PHASE 1, STEP 2: WAKE-WORD DETECTION")
    print("="*60)
    print("\nThis script will:")
    print("1. Listen for the wake word: 'Hey Mycroft'")
    print("2. Record your command when detected")
    print("3. Transcribe it to text")
    print("4. Print the result")
    print("\n" + "-"*60)
    
    # Initialize wake-word detector
    detector = WakeWordDetector(wake_word="hey_mycroft", sensitivity=0.5)
    
    while True:
        print("\n" + "="*60)
        print("Waiting for wake word...")
        print("="*60)
        
        # Listen for wake word (with 5 minute timeout)
        if detector.listen_for_wake_word(timeout_seconds=300):
            # Wake word detected! Now record the command
            print("\n" + "-"*60)
            print("Recording your command...")
            print("-"*60)
            
            text = record_and_transcribe(duration_seconds=10, model_name="base")
            
            if text:
                print("\n" + "="*60)
                print("RECOGNIZED COMMAND")
                print("="*60)
                print(f"You said: \"{text}\"")
                print("="*60)
            else:
                print("\n❌ Failed to transcribe command.")
        else:
            print("\nNo wake word detected. Exiting.")
            break


if __name__ == "__main__":
    main()