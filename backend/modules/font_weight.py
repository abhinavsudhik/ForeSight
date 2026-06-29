"""
Font Weight Inconsistency Detection for ForeSight.
Detects anomalous bolding or varying stroke weights in text characters of the document.
"""

import cv2
import numpy as np
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)

def analyze_font_weight(image_path: str, ocr_word_boxes: list = None) -> dict:
    """
    Analyze the stroke width consistency across text words in the document.
    
    This signal is WEAK for clean digital edit -> print jobs (uniform stroke width throughout).
    It is STRONG for mixed-source documents where text blocks come from different printers or fonts.
    Weight kept conservative at 0.60 max.
    
    Parameters
    ----------
    image_path : str
        Path to the document image file.
    ocr_word_boxes : list, optional
        List of (x, y, w, h, text) tuples from OCR. If None or empty,
        a contour-based heuristic word detector is used.
        
    Returns
    -------
    dict
        {
            "mean_stroke_width": float,
            "std_deviation": float,
            "outlier_count": int,
            "outlier_words": list[dict],
            "risk": "low" | "medium" | "high",
            "penalty_weight": float
        }
    """
    logger.info("Running font weight analysis on: %s", image_path)
    
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Failed to load image for font weight analysis: %s", image_path)
        return {
            "mean_stroke_width": 0.0,
            "std_deviation": 0.0,
            "outlier_count": 0,
            "outlier_words": [],
            "risk": "high",
            "penalty_weight": 0.60
        }
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    
    # Apply Otsu's thresholding (binary inversion assuming dark text on light background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Extract word boxes heuristically if None or empty
    if not ocr_word_boxes:
        logger.info("No OCR boxes provided. Using morphological word-blob extraction fallback.")
        # Dilation to group characters into words horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ocr_word_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter typical word dimensions on a standard document page
            if 8 <= h <= 100 and 15 <= w <= 500:
                ocr_word_boxes.append((x, y, w, h, "word"))
                
    stroke_widths = []
    words_metadata = []
    
    for item in ocr_word_boxes:
        # Support both (x, y, w, h, text) tuple and dictionary {"text": text, "box": [x, y, w, h]}
        if isinstance(item, dict):
            text = item.get("text", "")
            box = item.get("box", [0, 0, 0, 0])
            if len(box) == 4:
                x, y, w, h = box
            else:
                continue
        elif isinstance(item, (tuple, list)) and len(item) == 5:
            x, y, w, h, text = item
        else:
            continue
            
        if len(text) < 3:
            continue
            
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w_img, int(x + w))
        y1 = min(h_img, int(y + h))
        
        if x1 <= x0 or y1 <= y0:
            continue
            
        patch = binary[y0:y1, x0:x1]
        if np.sum(patch > 0) == 0:
            continue
            
        # Distance transform finds distance to nearest zero (background) pixel
        dist = ndimage.distance_transform_edt(patch)
        mean_width = float(np.mean(dist[patch > 0]) * 2)
        
        stroke_widths.append(mean_width)
        words_metadata.append({
            "word": text,
            "stroke_width": mean_width,
            "box": (x0, y0, x1 - x0, y1 - y0)
        })
        
    if not stroke_widths:
        return {
            "mean_stroke_width": 0.0,
            "std_deviation": 0.0,
            "outlier_count": 0,
            "outlier_words": [],
            "risk": "low",
            "penalty_weight": 0.0
        }
        
    mean_w = float(np.mean(stroke_widths))
    std_w = float(np.std(stroke_widths))
    
    outliers = []
    for item in words_metadata:
        z = (item["stroke_width"] - mean_w) / std_w if std_w > 0.0 else 0.0
        if abs(z) > 2.5:
            outliers.append({
                "word": item["word"],
                "stroke_width": round(item["stroke_width"], 3),
                "z_score": round(z, 3),
                "box": item["box"]
            })
            
    outlier_count = len(outliers)
    
    # Risk logic:
    # - outlier_count >= 3: risk "high", weight 0.60
    # - outlier_count == 1-2: risk "medium", weight 0.35
    # - outlier_count == 0: risk "low", weight 0.0
    if outlier_count >= 3:
        risk = "high"
        penalty_weight = 0.60
    elif outlier_count >= 1:
        risk = "medium"
        penalty_weight = 0.35
    else:
        risk = "low"
        penalty_weight = 0.0
        
    return {
        "mean_stroke_width": round(mean_w, 3),
        "std_deviation": round(std_w, 3),
        "outlier_count": outlier_count,
        "outlier_words": outliers,
        "risk": risk,
        "penalty_weight": penalty_weight
    }
