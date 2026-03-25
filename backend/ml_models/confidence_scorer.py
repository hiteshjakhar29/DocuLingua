import os
import re
import datetime
import cv2
import logging
from typing import Dict, Any, List

logger = logging.getLogger("ConfidenceScorer")

class ConfidenceScorer:
    """
    Evaluates extraction confidence for an OCR/NER field by combining multiple weighted factors.
    """
    
    def __init__(self):
        # Weight distribution
        self.weights = {
            "ocr_agreement": 0.30,
            "ocr_char_confidence": 0.25,
            "ner_probability": 0.25,
            "pattern_validation": 0.10,
            "image_quality": 0.10
        }
    
    def _validate_pattern(self, field_name: str, field_value: str) -> float:
        """
        Validates the extracted value against domain-specific patterns.
        Returns a pattern correlation score between 0.0 and 1.0.
        """
        if not field_value:
            return 0.0
            
        field_value = str(field_value).strip()
        field_name = field_name.upper()
        
        # DATE validation
        if field_name in ["DATE", "VALID_UNTIL"]:
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",     # YYYY-MM-DD
                r"\d{2}-\d{2}-\d{4}",     # DD-MM-YYYY
                r"\d{2}/\d{2}/\d{4}",     # MM/DD/YYYY or DD/MM/YYYY
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b"
            ]
            matched = any(re.search(pat, field_value, re.IGNORECASE) for pat in date_patterns)
            
            is_calendar_valid = False
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
                try:
                    datetime.datetime.strptime(field_value, fmt)
                    is_calendar_valid = True
                    break
                except ValueError:
                    continue
            
            if is_calendar_valid: return 1.0
            if matched: return 0.7 # Matched regex but couldn't parse directly
            return 0.2 # Unrecognized format
            
        # CERT_NUMBER validation
        elif field_name == "CERT_NUMBER":
            if re.match(r"^[A-Za-z0-9\-\_]{5,20}$", field_value):
                return 1.0
            elif re.search(r"[0-9]{4,}", field_value):
                return 0.6
            return 0.1
            
        # GRADE validation
        elif field_name == "GRADE":
            # Letter grades
            if re.match(r"^[A-F][\+\-]?$", field_value, re.IGNORECASE):
                return 1.0
            # GPA
            if re.match(r"^[0-4]\.\d{1,2}$", field_value):
                return 1.0
            # Percentage
            if re.match(r"^(100|[1-9]?[0-9])%$", field_value):
                return 1.0
            return 0.0
            
        # PERSON / NAME validation
        elif field_name in ["PERSON", "NAME"]:
            if any(char.isdigit() for char in field_value):
                return 0.1
                
            words = field_value.split()
            if all(word[0].isupper() for word in words if word):
                return 1.0
            if re.match(r"^[A-Za-z\s\'\-]+$", field_value):
                return 0.7
            return 0.3
            
        # INSTITUTION validation
        elif field_name == "INSTITUTION":
            keywords = ["university", "college", "institute", "school", "academy", "polytechnic"]
            score = 0.5
            if any(k in field_value.lower() for k in keywords):
                score += 0.3
            if not any(char.isdigit() for char in field_value):
                score += 0.2
            return min(score, 1.0)

        # Unknown field
        return 0.5
        
    def _calculate_image_quality(self, image_path: str = None, contrast: float = None, sharpness: float = None) -> float:
        """
        Calculates image quality from explicit parameters or by lazily loading cv2 arrays.
        """
        if contrast is not None and sharpness is not None:
            return min(1.0, max(0.0, (contrast + sharpness) / 2))
            
        if image_path and os.path.exists(image_path):
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return 0.5
                
                # Laplacian variance for sharpness
                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                computed_sharpness = min(1.0, lap_var / 500.0)
                
                # RMS contrast
                computed_contrast = min(1.0, img.std() / 128.0)
                
                return (computed_sharpness + computed_contrast) / 2.0
            except Exception as e:
                logger.warning(f"Image quality calculation failed: {e}")
                return 0.5
                
        return 0.5

    def calculate_field_confidence(self, field_name: str, field_value: str, ocr_data: Dict[str, Any], ner_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates OCR, NER, Pattern, and Image inputs into a unified 0-100% confidence score.
        """
        ocr_agree = ocr_data.get("agreement_percentage", 0.0)
        if ocr_agree > 1.0: 
            ocr_agree /= 100.0
            
        ocr_char = ocr_data.get("character_confidence", 0.0)
        ner_prob = ner_data.get("probability", 0.0)
        
        pattern_score = self._validate_pattern(field_name, field_value)
        
        img_q = ocr_data.get("image_quality_metrics", {})
        quality_score = self._calculate_image_quality(
            image_path=ocr_data.get("image_path"),
            contrast=img_q.get("contrast"),
            sharpness=img_q.get("sharpness")
        )
        
        weighted_score = (
            (ocr_agree * self.weights["ocr_agreement"]) +
            (ocr_char * self.weights["ocr_char_confidence"]) +
            (ner_prob * self.weights["ner_probability"]) +
            (pattern_score * self.weights["pattern_validation"]) +
            (quality_score * self.weights["image_quality"])
        )
        
        final_confidence = max(0.0, min(1.0, weighted_score))
        
        return {
            "field": field_name,
            "value": field_value,
            "confidence_score": round(final_confidence * 100, 2),
            "is_flagged": self.get_manual_review_flag(final_confidence * 100),
            "breakdown": {
                "ocr_agreement": round(ocr_agree, 3),
                "ocr_char_confidence": round(ocr_char, 3),
                "ner_probability": round(ner_prob, 3),
                "pattern_validation": round(pattern_score, 3),
                "image_quality": round(quality_score, 3)
            }
        }

    def get_manual_review_flag(self, confidence_score: float) -> bool:
        """
        Flags manual review boolean target if confidence dips under 70%.
        """
        return confidence_score < 70.0

    def generate_confidence_report(self, all_fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates extracted entity inputs into a hierarchical document validation report.
        """
        if not all_fields:
            return {"document_confidence": 0.0, "requires_manual_review": True, "fields": []}
            
        total_score = sum(f["confidence_score"] for f in all_fields)
        avg_score = round(total_score / len(all_fields), 2)
        
        requires_review = any(f["is_flagged"] for f in all_fields)
        
        return {
            "document_confidence_average": avg_score,
            "requires_manual_review": requires_review or (avg_score < 75.0),
            "flagged_fields_count": sum(1 for f in all_fields if f["is_flagged"]),
            "fields_breakdown": all_fields
        }

if __name__ == "__main__":
    scorer = ConfidenceScorer()
    
    # Simple self test
    dt_field = scorer.calculate_field_confidence(
        "DATE", "2024-12-05", 
        {"agreement_percentage": 98.0, "character_confidence": 0.95},
        {"probability": 0.99}
    )
    person_field = scorer.calculate_field_confidence(
        "PERSON", "john 123", 
        {"agreement_percentage": 60.0, "character_confidence": 0.50},
        {"probability": 0.40}
    )
    
    report = scorer.generate_confidence_report([dt_field, person_field])
    print(re.sub(r'([A-Za-z_]+)', r'\1', str(report)))
