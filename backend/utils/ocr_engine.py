import io
import logging
import os
import re
import shutil
import time
from typing import Any, Dict, Optional
import base64
from langchain_core.messages import HumanMessage

import cv2
import numpy as np
import pytesseract
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
from PIL import Image

logger = logging.getLogger(__name__)


def _configure_tesseract_cmd() -> str | None:
    """Resolve tesseract executable from env, PATH, or common Windows locations."""
    candidates = []

    env_cmd = (os.getenv("TESSERACT_CMD") or "").strip().strip('"')
    if env_cmd:
        candidates.append(env_cmd)

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        candidates.append(path_cmd)

    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return None


_TESSERACT_CMD = _configure_tesseract_cmd()

def _get_tesseract_langs() -> str:
    """Detect available languages in Tesseract. Defaults to eng+hin if possible."""
    try:
        langs = pytesseract.get_languages(config="")
        if "hin" in langs:
            return "eng+hin"
    except Exception:
        pass
    return "eng"

_TESSERACT_LANGS = _get_tesseract_langs()

def _deskew(binary_img: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary_img > 0))
    if len(coords) == 0:
        return binary_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _preprocess_variants_for_ocr(image_bytes: bytes) -> Dict[str, np.ndarray]:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale for small/low-resolution text.
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Noise removal + thresholding variants for uneven lighting/fonts.
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(clahe, -1, kernel)
    _, sharpened_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    variants = {
        "adaptive": adaptive,
        "otsu": otsu,
        "sharpened_otsu": sharpened_otsu,
        "morph_close": morph,
    }
    return {name: _deskew(im) for name, im in variants.items()}


def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0

    tokenized = [tok for tok in re.split(r"\s+", text) if tok]
    if not tokenized:
        return 0.0

    alnum_tokens = [tok for tok in tokenized if any(ch.isalnum() for ch in tok)]
    quality_ratio = len(alnum_tokens) / len(tokenized)
    avg_token_len = sum(len(tok) for tok in alnum_tokens) / max(len(alnum_tokens), 1)

    return min(1.0, quality_ratio) * 0.75 + min(avg_token_len / 6.0, 1.0) * 0.25


def _ocr_with_confidence(processed_img: np.ndarray, psm: int = 6, lang: str = "eng") -> Dict[str, Any]:
    data = pytesseract.image_to_data(
        processed_img,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config=f"--oem 3 --psm {psm} --dpi 300",
    )
    words = []
    confidences = []

    for txt, conf in zip(data["text"], data["conf"]):
        cleaned = (txt or "").strip()
        if not cleaned:
            continue
        words.append(cleaned)
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = -1
        # For Hindi, alphanumeric check might be too strict, but we'll keep it for now
        if conf_val >= 20:
            confidences.append(conf_val)

    text = " ".join(words).strip()
    confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    quality = _text_quality_score(text)
    return {
        "text": text,
        "confidence": round(confidence, 2),
        "quality": round(quality, 4),
    }


def extract_text_via_gemini(image_bytes: bytes) -> str:
    """
    Multimodal OCR using Gemini.
    Converts image to base64 and queries Gemini Vision model.
    """
    try:
        from backend.llm_manager import get_fallback_llm
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Identify and transcribe all text present in this image precisely. "
                            "Preserve spacing, lines, and formatting where possible. "
                            "Do not include any chat commentary or conversational filler—only output the transcribed text."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
        
        llm = get_fallback_llm()
        response = llm.invoke([message])
        extracted_text = getattr(response, "content", str(response)).strip()
        return extracted_text
    except Exception as e:
        logger.error(f"Gemini Vision OCR failed: {e}")
        raise e


def extract_text_from_image(image_file, language: str = "english") -> Dict[str, Any]:
    """
    OCR with preprocessing, confidence score, and retry fallback.
    Supports English and Hindi.
    """
    t_start = time.perf_counter()
    
    # Map to tesseract lang codes
    tess_lang = "hin+eng" if language.lower() == "hindi" and "hin" in _TESSERACT_LANGS else "eng"
    if "hin" not in tess_lang and language.lower() == "hindi":
        logger.warning("Hindi language pack not found in Tesseract, falling back to English.")

    try:
        if isinstance(image_file, (bytes, bytearray)):
            image_bytes = bytes(image_file)
        elif isinstance(image_file, str):
            with open(image_file, "rb") as f:
                image_bytes = f.read()
        elif hasattr(image_file, "getvalue"):
            image_bytes = image_file.getvalue()
        elif hasattr(image_file, "read"):
            image_bytes = image_file.read()
        else:
            raise ValueError("Unsupported image input type for OCR.")

        logger.debug("OCR input: %d bytes, language: %s", len(image_bytes), tess_lang)

        if not _TESSERACT_CMD:
            logger.info("Tesseract executable not found. Routing OCR task to Gemini Vision.")
            text = extract_text_via_gemini(image_bytes)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                "text": text,
                "confidence": 100.0,
                "method": "gemini_multimodal",
                "error": None,
                "processing_time_ms": round(elapsed_ms, 1),
            }

        variants = _preprocess_variants_for_ocr(image_bytes)
        
        best = {
            "text": "",
            "confidence": 0.0,
            "quality": 0.0,
            "method": "none",
            "score": -1.0,
        }

        # Try fast path
        fast_candidates = [("otsu", 6), ("adaptive", 6)]
        for variant_name, psm in fast_candidates:
            candidate = _ocr_with_confidence(variants[variant_name], psm=psm, lang=tess_lang)
            text = candidate.get("text", "").strip()
            confidence = candidate.get("confidence", 0.0)
            
            if text and confidence >= 80:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                return {
                    "text": text,
                    "confidence": confidence,
                    "method": f"{variant_name}_psm{psm}_fast",
                    "error": None,
                    "processing_time_ms": round(elapsed_ms, 1),
                }

        # Full sweep if fast path fails
        for variant_name, processed in variants.items():
            for psm in (6, 11):
                candidate = _ocr_with_confidence(processed, psm=psm, lang=tess_lang)
                text = candidate.get("text", "").strip()
                confidence = candidate.get("confidence", 0.0)
                quality = candidate.get("quality", 0.0)

                score = confidence * 0.7 + quality * 30
                if score > best["score"]:
                    best = {
                        "text": text,
                        "confidence": confidence,
                        "method": f"{variant_name}_psm{psm}",
                        "score": score
                    }

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return {
            "text": best["text"],
            "confidence": best["confidence"],
            "method": best["method"],
            "error": None,
            "processing_time_ms": round(elapsed_ms, 1),
        }

    except Exception as exc:
        try:
            logger.info("Tesseract execution failed or not available. Falling back to Gemini Vision OCR...")
            # If image_bytes is not defined yet (e.g. read error), we try to extract it from image_file
            if 'image_bytes' not in locals():
                if isinstance(image_file, (bytes, bytearray)):
                    image_bytes = bytes(image_file)
                elif isinstance(image_file, str):
                    with open(image_file, "rb") as f:
                        image_bytes = f.read()
                elif hasattr(image_file, "getvalue"):
                    image_bytes = image_file.getvalue()
                elif hasattr(image_file, "read"):
                    image_bytes = image_file.read()
                else:
                    raise ValueError("Unsupported image input type for OCR.")
            text = extract_text_via_gemini(image_bytes)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                "text": text,
                "confidence": 95.0,
                "method": "gemini_vision_fallback",
                "error": None,
                "processing_time_ms": round(elapsed_ms, 1),
            }
        except Exception as gemini_exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                "text": "",
                "confidence": 0.0,
                "method": "error",
                "error": f"Tesseract error: {exc}. Gemini fallback error: {gemini_exc}",
                "processing_time_ms": round(elapsed_ms, 1),
            }

def is_page_scanned(text: str, threshold: int = 50) -> bool:
    printable = (text or "").strip()
    return len(printable) < threshold

def extract_text_from_pdf_page(page_image_bytes: bytes, language: str = "english") -> Dict[str, Any]:
    return extract_text_from_image(page_image_bytes, language=language)
