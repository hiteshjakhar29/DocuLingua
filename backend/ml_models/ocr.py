import cv2
import numpy as np
import pytesseract
import easyocr
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCREngine")

class OCREngine:
    """
    A unified OCR Engine supporting Tesseract and EasyOCR.
    Handles image preprocessing, dual execution, language detection, and output merging.
    """
    
    # Mapping of user-facing language codes to EasyOCR and Tesseract codes
    LANG_MAP = {
        'en': {'tess': 'eng', 'easy': 'en'},
        'ar': {'tess': 'ara', 'easy': 'ar'},
        'es': {'tess': 'spa', 'easy': 'es'},
        'fr': {'tess': 'fra', 'easy': 'fr'},
        'hi': {'tess': 'hin', 'easy': 'hi'},
        'de': {'tess': 'deu', 'easy': 'de'},
        'pt': {'tess': 'por', 'easy': 'pt'},
        'zh': {'tess': 'chi_sim', 'easy': 'ch_sim'},
        'ru': {'tess': 'rus', 'easy': 'ru'},
        'ur': {'tess': 'urd', 'easy': 'ur'}
    }

    def __init__(self):
        """
        Initializes the OCR Engine and caches a default English EasyOCR reader.
        """
        self._easy_readers = {}
        # Pre-initialize English since it's most common
        self._get_easyocr_reader(['en'])
        
    def _get_easyocr_reader(self, langs: List[str]):
        """
        Dynamically handles incompatible EasyOCR language combinations 
        (e.g., Arabic and Chinese can't be loaded together).
        """
        # Sort to create a consistent cache key
        langs = sorted(langs)
        key = "_".join(langs)
        if key not in self._easy_readers:
            try:
                easy_langs = [self.LANG_MAP.get(l, {}).get('easy', 'en') for l in langs]
                # Default to 'en' if completely unknown
                if not easy_langs:
                    easy_langs = ['en']
                logger.info(f"Initializing EasyOCR reader for languages: {easy_langs}")
                self._easy_readers[key] = easyocr.Reader(easy_langs, gpu=False)
            except ValueError as e:
                logger.warning(f"Incompatible language combination {easy_langs}: {e}. Falling back to English.")
                self._easy_readers[key] = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR for {langs}: {e}")
                raise
        return self._easy_readers[key]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocesses the image for OCR: Grayscale, Deskew, Denoise, Contrast, Binarize, Border Removal.
        """
        logger.info(f"Preprocessing image: {image_path}")
        try:
            # 1. Load Image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")

            # 2. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrasted = clahe.apply(gray)

            # 4. Noise reduction (Gaussian Blur)
            blurred = cv2.GaussianBlur(contrasted, (5, 5), 0)

            # 5. Adaptive Thresholding (Binarization)
            binarized = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # 6. Deskewing using Hough Transform
            edges = cv2.Canny(binarized, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            
            angle = 0.0
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                
                # Use median angle to ignore outliers
                median_angle = np.median(angles)
                # Only correct small skews (e.g. within -45 to 45 degrees)
                if -45 <= median_angle <= 45:
                    angle = median_angle

            if abs(angle) > 0.5:
                (h, w) = binarized.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binarized = cv2.warpAffine(binarized, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                logger.debug(f"Deskewed image by {angle:.2f} degrees")

            # 7. Border removal
            # Find contours to crop to the largest bounding box of content
            contours, _ = cv2.findContours(cv2.bitwise_not(binarized), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h_rect = cv2.boundingRect(c)
                # Leave a small margin
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(binarized.shape[1] - x, w + 2*margin)
                h_rect = min(binarized.shape[0] - y, h_rect + 2*margin)
                binarized = binarized[y:y+h_rect, x:x+w]

            return binarized

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def extract_text_tesseract(self, image: np.ndarray, langs: List[str] = ['en']) -> Dict[str, Any]:
        """
        Runs Tesseract OCR. If language detection is required, it can be done beforehand.
        """
        logger.info(f"Running Tesseract OCR for languages: {langs}")
        try:
            tess_langs = [self.LANG_MAP.get(l, {}).get('tess', 'eng') for l in langs]
            lang_str = "+".join(tess_langs)
            
            # Use PSM 3 (Fully automatic page segmentation, but no OSD)
            custom_config = f'--oem 3 --psm 3 -l {lang_str}'
            
            # Extract detailed data
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            words = []
            confidences = []
            
            text_blocks = []
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                try:
                    conf = float(data['conf'][i])
                except ValueError:
                    conf = -1.0
                
                if word and conf > 0:
                    words.append(word)
                    confidences.append(conf)
                    text_blocks.append(word)
            
            full_text = " ".join(text_blocks)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "text": full_text,
                "confidence": min(max(avg_conf / 100.0, 0.0), 1.0) # Normalize to 0-1
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {"text": "", "confidence": 0.0}

    def extract_text_easyocr(self, image: np.ndarray, langs: List[str] = ['en']) -> Dict[str, Any]:
        """
        Runs EasyOCR with multi-language support.
        """
        logger.info(f"Running EasyOCR for languages: {langs}")
        try:
            reader = self._get_easyocr_reader(langs)
            results = reader.readtext(image)
            
            text_blocks = []
            confidences = []
            
            for (bbox, text, prob) in results:
                if text.strip():
                    text_blocks.append(text)
                    confidences.append(prob)
            
            full_text = " ".join(text_blocks)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "text": full_text,
                "confidence": avg_conf
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {"text": "", "confidence": 0.0}

    def calculate_agreement(self, text1: str, text2: str) -> float:
        """
        Calculates similarity between two texts using SequenceMatcher.
        """
        if not text1 and not text2:
             return 1.0
        matcher = SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
        
    def _merge_texts(self, tesseract_res: Dict, easyocr_res: Dict) -> str:
        """
        Simple merging heuristic: return the string with higher confidence, 
        or prefer EasyOCR if it picked up significantly more structure.
        """
        # If one is empty, return the other
        if not tesseract_res['text']: return easyocr_res['text']
        if not easyocr_res['text']: return tesseract_res['text']
        
        # Here we just select the one with the higher engine confidence.
        # A more sophisticated approach would be word-by-word merging or positional merging
        if easyocr_res['confidence'] > tesseract_res['confidence']:
            return easyocr_res['text']
        return tesseract_res['text']

    def run_dual_ocr(self, image_path: str, langs: List[str] = ['en']) -> Dict[str, Any]:
        """
        Executes the full pipeline: Preprocessing -> Parallel Dual OCR -> Aggregation.
        """
        start_time = time.time()
        logger.info(f"Starting Dual OCR pipeline for {image_path}")

        try:
            # 1. Preprocess
            processed_img = self.preprocess_image(image_path)
            
            # 2. Parallel OCR execution
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_tess = executor.submit(self.extract_text_tesseract, processed_img, langs)
                future_easy = executor.submit(self.extract_text_easyocr, processed_img, langs)
                
                tess_result = future_tess.result()
                easy_result = future_easy.result()
                
            # 3. Analyze agreement
            agreement = self.calculate_agreement(tess_result['text'], easy_result['text'])
            
            # 4. Merge results
            merged_text = self._merge_texts(tess_result, easy_result)
            
            # Character confidence (using the winning engine's confidence as proxy)
            char_conf = max(tess_result['confidence'], easy_result['confidence'])
            
            processing_time = time.time() - start_time
            logger.info(f"Dual OCR complete in {processing_time:.2f}s. Agreement: {agreement:.2f}")

            return {
                "extracted_text": merged_text,
                "tesseract_text": tess_result['text'],
                "easyocr_text": easy_result['text'],
                "character_confidence_scores": {
                    "tesseract_avg": tess_result['confidence'],
                    "easyocr_avg": easy_result['confidence'],
                    "overall_confidence": char_conf
                },
                "agreement_percentage": round(agreement * 100, 2),
                "processing_time": round(processing_time, 2)
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2)
            }

if __name__ == "__main__":
    # Simple self-test code when run directly
    import tempfile
    engine = OCREngine()
    
    # Create a quick dummy image with some text
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
        img = np.zeros((300, 600, 3), dtype=np.uint8) + 255
        cv2.putText(img, "Test OCR English Phrase", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
        cv2.imwrite(tmp.name, img)
        
        result = engine.run_dual_ocr(tmp.name, langs=['en'])
        print("\n=== OCR RESULT ===")
        print(f"Extracted Text: {result.get('extracted_text')}")
        print(f"Agreement: {result.get('agreement_percentage')}%")
        print(f"Time: {result.get('processing_time')}s")
