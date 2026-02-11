import contextlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from ultralytics import YOLO

# ── Compiled patterns (shared across all instances) ─────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EMOJI_RE = re.compile(
    r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F6FF"
    r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]"
)
_IMAGE_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


# ── Small helpers ───────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Strip ANSI escapes, emoji, and carriage returns."""
    text = _ANSI_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text.replace("\r", "\n")


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _getattr_chain(obj: Any, *attrs: str) -> Optional[float]:
    """Safely traverse nested attributes: _getattr_chain(box, 'mp') -> box.mp."""
    for a in attrs:
        if obj is None or not hasattr(obj, a):
            return None
        obj = getattr(obj, a)
    return obj


def _sum_values(value: Any) -> Optional[float]:
    """Sum a numerical array / tensor, return float or None."""
    if value is None:
        return None
    try:
        total = value.sum()
        return float(total.item() if hasattr(total, "item") else total)
    except Exception:
        pass
    try:
        return float(sum(value))
    except Exception:
        return None


# ── Dataset I/O ─────────────────────────────────────────────────

def _read_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _scan_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]


def _read_image_list(list_path: Path, base: Path) -> List[Path]:
    images: List[Path] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                images.append(_resolve(base, line))
    return images


def _collect_val_images(data_path: str) -> List[Path]:
    yaml_path = Path(data_path)
    data = _read_yaml(data_path)
    base = Path(data.get("path") or yaml_path.parent)
    val = data.get("val") or data.get("test")
    if not val:
        raise ValueError("Dataset yaml must include 'val' or 'test' path.")

    images: List[Path] = []
    for item in (val if isinstance(val, (list, tuple)) else [val]):
        p = _resolve(base, item)
        if p.is_dir():
            images.extend(_scan_images(p))
        elif p.is_file() and p.suffix.lower() == ".txt":
            images.extend(_read_image_list(p, base))
        else:
            images.append(p)
    return images


# ── Label / box utilities ───────────────────────────────────────

def _label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError:
        pass
    return Path(*parts).with_suffix(".txt")


def _load_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                labels.append((int(float(parts[0])), *map(float, parts[1:5])))
    return labels


def _xywhn_to_xyxy(
    labels: List[Tuple[int, float, float, float, float]], w: int, h: int,
) -> np.ndarray:
    if not labels:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.array(labels, dtype=np.float32)
    cx, cy = arr[:, 1] * w, arr[:, 2] * h
    bw, bh = arr[:, 3] * w, arr[:, 4] * h
    return np.stack(
        [arr[:, 0], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1,
    )


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0)


# ── Logging infrastructure ──────────────────────────────────────

class _LogRedirect:
    """Intercepts stdout / stderr, sanitizes text, forwards to callback.

    Always instantiated (even with callback=None) so PyInstaller's
    console=False mode doesn't crash on None stdout/stderr.
    """

    def __init__(self, callback=None):
        self._cb = callback

    def write(self, text):
        if not text:
            return
        try:
            cleaned = _sanitize(text)
            if self._cb:
                self._cb(cleaned)
        except Exception:
            pass

    def flush(self):
        pass

    def isatty(self):
        return False


class _CallbackLogHandler(logging.Handler):
    """Logging handler that strips emoji and forwards to a callback."""

    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def emit(self, record):
        if not self._cb:
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if msg:
            self._cb(_EMOJI_RE.sub("", msg) + "\n")


@contextlib.contextmanager
def _log_context(callback, verbose: bool):
    """Combined context manager: redirect stdout/stderr + optional log handler."""
    # Forward stdout/stderr to UI only when verbose; otherwise just sink it
    # (still need a redirect for PyInstaller console=False safety)
    redirect = _LogRedirect(callback if verbose else None)
    # Set up log handler only when verbose
    if verbose and callback:
        handler = _CallbackLogHandler(callback)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        ultra = logging.getLogger("ultralytics")
        prev_level, prev_prop = ultra.level, ultra.propagate
        root.addHandler(handler)
        ultra.setLevel(logging.INFO)
        ultra.propagate = True
    else:
        handler = None

    try:
        with contextlib.redirect_stdout(redirect), contextlib.redirect_stderr(redirect):
            yield
    finally:
        if handler:
            logging.getLogger().removeHandler(handler)
            ultra = logging.getLogger("ultralytics")
            ultra.setLevel(prev_level)        # type: ignore[possibly-undefined]
            ultra.propagate = prev_prop     # type: ignore[possibly-undefined]


# ── TP / FP fallback computation ────────────────────────────────

def _compute_tp_fp(
    model: YOLO,
    data_path: str,
    log_callback=None,
    verbose: bool = False,
    is_onnx: bool = False,
) -> Tuple[int, int]:
    images = _collect_val_images(data_path)
    if log_callback:
        log_callback(f"Computing TP/FP on {len(images)} images...\n")

    predict_kw: dict = {"conf": 0.25, "iou": 0.5, "verbose": verbose}
    if is_onnx:
        model.overrides["batch"] = 1
        predict_kw["batch"] = 1
        if log_callback:
            log_callback("ONNX: batch=1 for prediction.\n")

    with _log_context(log_callback, verbose):
        if is_onnx:
            results = []
            for img in images:
                pred = model.predict(source=str(img), **predict_kw)
                if pred:
                    results.append(pred[0])
        else:
            results = model.predict(source=[str(p) for p in images], **predict_kw)

    tp = fp = total_gt = total_preds = missing_labels = 0

    for img_path, result in zip(images, results):
        h, w = result.orig_shape
        label_path = _label_path_for(img_path)
        if not label_path.exists():
            missing_labels += 1
        gt = _xywhn_to_xyxy(_load_labels(label_path), w, h)
        gt_used = np.zeros(gt.shape[0], dtype=bool)
        total_gt += gt.shape[0]

        if result.boxes is None or result.boxes.xyxy is None:
            continue

        pred_xyxy = result.boxes.xyxy.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        total_preds += len(pred_xyxy)

        conf = getattr(result.boxes, "conf", None)
        order = (
            np.argsort(-conf.cpu().numpy()) if conf is not None
            else range(len(pred_xyxy))
        )

        for idx in order:
            if gt.shape[0] == 0:
                fp += 1
                continue
            mask = (gt[:, 0].astype(int) == pred_cls[idx]) & ~gt_used
            if not mask.any():
                fp += 1
                continue
            ious = _box_iou(pred_xyxy[idx], gt[mask][:, 1:5])
            best = np.argmax(ious)
            if ious[best] >= 0.5:
                tp += 1
                gt_used[np.where(mask)[0][best]] = True
            else:
                fp += 1

    if log_callback:
        log_callback(
            f"TP/FP summary: tp={tp}, fp={fp}, images={len(images)}, "
            f"gt={total_gt}, preds={total_preds}, missing_labels={missing_labels}\n"
        )
        if total_gt > 0 and total_preds > 0 and tp == 0:
            log_callback(
                "Warning: TP=0 with GT and predictions present. "
                "This often means model/dataset class mismatch.\n"
            )

    return tp, fp


# ── Metric extraction ───────────────────────────────────────────

def _extract_metrics(results) -> Dict[str, Optional[float]]:
    box = getattr(results, "box", None)

    precision = _getattr_chain(box, "mp")
    if precision is None:
        precision = _getattr_chain(box, "p")

    recall = _getattr_chain(box, "mr")
    if recall is None:
        recall = _getattr_chain(box, "r")

    map50 = _getattr_chain(box, "map50")
    map50_95 = _getattr_chain(box, "map")

    speed = getattr(results, "speed", None)
    speed_ms = speed.get("inference") if isinstance(speed, dict) else None

    tp = fp = None
    cm = getattr(results, "confusion_matrix", None)
    if cm is not None and hasattr(cm, "matrix") and cm.matrix is not None:
        m = cm.matrix
        if m.size >= 4:
            tp_val = int(m[:-1, :-1].diagonal().sum())
            fp_val = int(m[-1, :-1].sum())
            if tp_val + fp_val > 0:
                tp, fp = tp_val, fp_val

    if tp is None:
        v = _sum_values(getattr(box, "tp", None))
        if v is not None:
            tp = int(round(v))
    if fp is None:
        v = _sum_values(getattr(box, "fp", None))
        if v is not None:
            fp = int(round(v))

    return {
        "speed_ms": speed_ms,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "tp": tp,
        "fp": fp,
    }


# ── Verbose environment log ─────────────────────────────────────

def _log_environment(model_path: str, data_path: str, cb):
    dataset = _read_yaml(data_path)
    names = dataset.get("names")
    n_cls = len(names) if isinstance(names, (list, dict)) else "n/a"

    cb("Extra logs enabled.\n")
    cb(f"Python: {sys.version.split()[0]}\n")
    try:
        import ultralytics as _ul
        cb(f"Ultralytics: {_ul.__version__}\n")
    except Exception:
        cb("Ultralytics: unknown\n")
    try:
        import torch
        cb(f"Torch: {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})\n")
    except Exception:
        cb("Torch: unknown\n")
    cb(f"Model: {model_path}\n")
    cb(f"Dataset: {data_path}\n")
    cb(f"Dataset base: {dataset.get('path', 'n/a')}\n")
    cb(f"Dataset val: {dataset.get('val', dataset.get('test', 'n/a'))}\n")
    cb(f"Classes: {n_cls}\n")
    cb("Validation: iou=0.5 | TP/FP: conf=0.25, iou=0.5\n")


# ── Public API ──────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    data_path: str,
    log_callback=None,
    verbose_logs: bool = False,
) -> Dict[str, Optional[float]]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    is_onnx = model_path.lower().endswith(".onnx")

    if log_callback:
        log_callback("Loading model...\n")
    if verbose_logs and log_callback:
        _log_environment(model_path, data_path, log_callback)

    with _log_context(log_callback, verbose_logs):
        model = YOLO(model_path, task="detect") if is_onnx else YOLO(model_path)

        val_kw: dict = {
            "data": data_path,
            "iou": 0.5,
            "plots": False,
            "save_json": False,
            "verbose": verbose_logs,
        }
        if is_onnx:
            model.overrides["batch"] = 1
            val_kw["batch"] = 1
            if log_callback:
                log_callback("ONNX: batch=1 for validation.\n")

        if log_callback:
            log_callback("Running validation...\n")
        results = model.val(**val_kw)

    if log_callback:
        log_callback("Collecting metrics...\n")

    metrics = _extract_metrics(results)

    if metrics.get("tp") is None or metrics.get("fp") is None:
        if log_callback:
            log_callback("TP/FP not provided by validator, running extra pass...\n")
        tp, fp = _compute_tp_fp(
            model, data_path, log_callback, verbose_logs, is_onnx,
        )
        metrics["tp"] = tp
        metrics["fp"] = fp

    return metrics
