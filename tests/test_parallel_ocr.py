"""start_ocr with a worker pool gives the same results as the serial path, and runs the
CPU-only engine off the calling thread."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bookpecker import pipeline
from bookpecker.config import PageConfig
from bookpecker.layout.base import Box, PageLayout, Region
from bookpecker.ocr.base import OcrLine, OcrResult
from bookpecker.ocr.ensemble import EnsembleEngine


class Fake:
    def __init__(self, name, text, cpu_only=False):
        self.name, self.text, self.cpu_only = name, text, cpu_only
        self.threads = set()

    def recognize(self, image, kind):
        self.threads.add(threading.current_thread().name)
        return OcrResult(self.name, [OcrLine(f"{self.text} {image.shape[0]}", [0, 0, 100, 20])])


def _layout():
    regions = [Region(Box(0, i * 200, 800, i * 200 + 100 + i), "text", id=i, order=i) for i in range(4)]
    return PageLayout(1, "right", 1000, 1000, Box(0, 0, 1000, 1000), regions)


def test_parallel_matches_serial(monkeypatch):
    gpu, tess = Fake("p", "alma"), Fake("t", "alma –", cpu_only=True)
    engine = EnsembleEngine(gpu, tess, known=None)
    monkeypatch.setattr(pipeline, "get_engine", lambda name, cfg: engine)
    gray = np.full((1000, 1000), 255, np.uint8)
    cfg = PageConfig()

    serial = pipeline.run_ocr(gray, _layout(), cfg)
    with ThreadPoolExecutor(3, thread_name_prefix="tess") as pool:
        finish = pipeline.start_ocr(gray, _layout(), cfg, pool)
        parallel = finish()

    assert serial == parallel and len(parallel) == 4
    assert gpu.threads == {threading.current_thread().name}
    assert any(t.startswith("tess") for t in tess.threads)
