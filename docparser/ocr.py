"""扫描页 OCR：RapidOCR(ONNX) 识别 + pypdfium2 栅格化。

设计约束：
- 懒加载单例引擎（模型随 wheel 内置，离线可用，首次调用约 1s 初始化）
- KB_OCR=0 时 ocr_enabled() 返回 False，解析器保持"显式失败"旧行为
- 只对无文本层的页面触发，文本页零开销
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

_engine = None
_lock = threading.Lock()


def ocr_enabled() -> bool:
    return os.getenv("KB_OCR", "1").strip().lower() not in {"0", "false", "off"}


def get_engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR()
    return _engine


def ocr_pdf_page(path: Path | str, page_index: int, dpi: int = 200,
                 min_score: float = 0.5) -> str:
    """栅格化 PDF 单页并返回按阅读顺序拼接的 OCR 文本（可能为空串）。"""
    import numpy as np
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[page_index]
        pil_image = page.render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        pdf.close()

    # RapidOCR 约定输入为 BGR ndarray
    array = np.asarray(pil_image)[:, :, ::-1]
    result, _elapse = get_engine()(array)
    if not result:
        return ""
    lines = [text for _box, text, score in result
             if float(score) >= min_score and text.strip()]
    return "\n".join(lines)
