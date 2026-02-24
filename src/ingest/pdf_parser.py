from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List
import pdfplumber
import pandas as pd  

logger = logging.getLogger(__name__)

def _table_to_markdown(table: List[List[str]]) -> str:
    #Convert a pdfplumber table to a clean, structured markdown string.
    if not table or not any(table):
        return ""
    
    df = pd.DataFrame(table)
    # Remove empty rows/cols often found in PDFs
    df = df.dropna(how='all').fillna("")
    
    # If the first row looks like a header, set it
    if len(df) > 1:
        df.columns = df.iloc[0]
        df = df[1:]
    
    return df.to_markdown(index=False)

def _detect_box_regions(page) -> List[str]:
    """Cluster words into boxes based on spatial proximity."""
    boxes = []
    try:
        words = page.words
        if not words or len(words) < 3: return boxes
        
        processed = set()
        for i, word1 in enumerate(words):
            if i in processed: continue
            box_words = [word1]
            x0_min, y0_min = word1['x0'], word1['top']
            x1_max, y1_max = word1['x1'], word1['bottom']
            
            for j, word2 in enumerate(words[i+1:], start=i+1):
                if j in processed: continue
                margin = 30
                overlap_x = not (word2['x1'] < x0_min - margin or word2['x0'] > x1_max + margin)
                overlap_y = not (word2['bottom'] < y0_min - margin or word2['top'] > y1_max + margin)
                
                if overlap_x and overlap_y:
                    box_words.append(word2)
                    processed.add(j)
                    x0_min, y0_min = min(x0_min, word2['x0']), min(y0_min, word2['top'])
                    x1_max, y1_max = max(x1_max, word2['x1']), max(y1_max, word2['bottom'])
            
            if len(box_words) >= 3:
                sorted_words = sorted(box_words, key=lambda w: (w['top'], w['x0']))
                box_text = " ".join(w['text'] for w in sorted_words if w.get('text', '').strip())
                if len(box_text) > 20:
                    # CRITICAL: Prefix for E5 Model
                    boxes.append(f"passage: {box_text.strip()}")
    except Exception as exc:
        logger.debug("Box detection error: %s", exc)
    return boxes

def parse_pdf(pdf_path: Path) -> List[Dict]:
    """Parse PDF into elements with E5 'passage:' prefixes."""
    elements = []
    pdf_path = Path(pdf_path)
    if not pdf_path.exists(): return elements

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # 1. Boxes
                for box_text in _detect_box_regions(page):
                    elements.append({"type": "box", "page_number": page_idx, "content": box_text})

                # 2. General Text
                text = page.extract_text() or ""
                if text.strip():
                    elements.append({"type": "text", "page_number": page_idx, "content": f"passage: {text.strip()}"})

                # 3. Tables
                for table in (page.extract_tables() or []):
                    md = _table_to_markdown(table)
                    if md.strip():
                        elements.append({"type": "table", "page_number": page_idx, "content": f"passage: {md}"})
    except Exception as exc:
        logger.error("Global parse error: %s", exc)
    return elements