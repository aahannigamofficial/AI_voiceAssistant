"""
Phase 1, Complete Pipeline Test

This script tests the full pipeline:
1. Listen for wake word ("Hey Mycroft")
2. Record the command
3. Transcribe to text
4. Recognize intent
5. Execute action

Run with: python test_full_pipeline.py
"""

from app.wake_word import WakeWordDetector
from app.speech import record_and_transcribe
from app.agent import IntentRecognizer
from app.computer import ComputerExecutor


def main():
    print("\n" + "="*60)
    print("PHASE 1: FULL PIPELINE TEST")
    print("="*60)
    print("\nThis script will:")
    print("1. Listen for 'Hey Mycroft'")
    print("2. Record your command")
    print("3. Transcribe it")
    print("4. Recognize your intent")
    print("5. Execute the action")
    print("\nExample commands:")
    print("  - 'Open YouTube'")
    print("  - 'Launch VS Code'")
    print("  - 'Search Google for machine learning'")
    print("  - 'Type hello world'")
    print("\n" + "-"*60)
    
    # Initialize components
    detector = WakeWordDetector(wake_word="hey_mycroft", sensitivity=0.5)
    recognizer = IntentRecognizer()
    executor = ComputerExecutor(enable_execution=True)  # Set to False for dry-run
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Listening for wake word...")
        print(f"{'='*60}")

        # Step 1: Listen for wake word
        if not detector.listen_for_wake_word(timeout_seconds=60):
            print("\nNo wake word detected. Exiting.")
            break

        # Step 2: Record command
        print("\n" + "-"*60)
        print("Recording your command (10 seconds)...")
        print("-"*60)

        transcribed_text = record_and_transcribe(duration_seconds=10, model_name="base")

        if not transcribed_text:
            print("\nFailed to transcribe. Waiting for the next wake word.")
            continue

        print(f"\nTranscribed: \"{transcribed_text}\"")

        # Step 3: Recognize intent
        print("\n" + "-"*60)
        print("Recognizing intent...")
        print("-"*60)

        intent = recognizer.recognize(transcribed_text)

        if not intent:
            print(f"Could not recognize intent from: \"{transcribed_text}\"")
            print("Waiting for the next wake word.")
            continue

        print(f"Recognized intent: {intent}")
        print(f"Confidence: {intent.confidence * 100:.0f}%")

        # Step 4: Execute action
        print("\n" + "-"*60)
        print("Executing action...")
        print("-"*60)

        success = executor.execute_intent(intent)

        if success:
            print("\nAction completed successfully!")
        else:
            print("\nAction failed!")


if __name__ == "__main__":
    main()