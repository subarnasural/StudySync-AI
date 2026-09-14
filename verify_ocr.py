import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging to see details of the OCR execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

# Load environment variables
load_dotenv()

# Add project root to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.utils.ocr_engine import extract_text_from_image

def main():
    print("=" * 60)
    print("AI Teaching Assistant - OCR Verification Script")
    print("=" * 60)

    image_path = "test_ocr.png"
    if not os.path.exists(image_path):
        print(f"Error: test_ocr.png not found at {os.path.abspath(image_path)}")
        sys.exit(1)

    print(f"Running OCR on image: {image_path}")
    print("Testing OCR engine (will use Gemini Vision if Tesseract is not installed)...")
    print("-" * 60)

    try:
        result = extract_text_from_image(image_path)
        print("\nOCR Execution Successful!")
        print("-" * 60)
        print(f"Extracted Text   :\n{result.get('text')}")
        print("-" * 60)
        print(f"Confidence       : {result.get('confidence')}%")
        print(f"Method Used      : {result.get('method')}")
        print(f"Processing Time  : {result.get('processing_time_ms')} ms")
        print(f"Error            : {result.get('error')}")
        print("=" * 60)
    except Exception as e:
        print(f"\nCRITICAL: OCR execution raised an unhandled exception: {e}")
        print("=" * 60)

if __name__ == "__main__":
    main()
