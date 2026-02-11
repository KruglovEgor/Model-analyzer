import contextlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from ultralytics import YOLO


def _safe_get(obj: Any, *attrs: str, default: Optional[float] = None) -> Optional[float]:
    current = obj
    for attr in attrs:
        if current is None or not hasattr(current, attr):
            return default
        current = getattr(current, attr)
    return current


def _sum_values(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        total = value.sum()
        if hasattr(total, "item"):
            return float(total.item())
        return float(total)
    except Exception:
        pass
    try:
        return float(sum(value))
    except Exception:
        pass
    try:
        return float(sum(sum(row) for row in value))
    except Exception:
        return None


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _collect_val_images(data_path: str) -> List[Path]:
    yaml_path = Path(data_path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    base = Path(data.get("path") or yaml_path.parent)
    val = data.get("val") or data.get("test")
    if not val:
        raise ValueError("Dataset yaml must include 'val' or 'test' path.")

    val_items: Iterable[str]
    if isinstance(val, (list, tuple)):
        val_items = val
    else:
        val_items = [val]

    images: List[Path] = []
    for item in val_items:
        path = _resolve_path(base, item)
        if path.is_dir():
            images.extend(_scan_images(path))
        elif path.is_file() and path.suffix.lower() == ".txt":
            images.extend(_read_image_list(path, base))
        else:
            images.append(path)

    return images


def _read_dataset_yaml(data_path: str) -> Dict[str, Any]:
    yaml_path = Path(data_path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _scan_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def _read_image_list(list_path: Path, base: Path) -> List[Path]:
    images: List[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            images.append(_resolve_path(base, line))
    return images


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        pass
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def _load_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            labels.append((cls, x, y, w, h))
    return labels


def _xywhn_to_xyxy(labels: List[Tuple[int, float, float, float, float]], w: int, h: int) -> np.ndarray:
    if not labels:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.array(labels, dtype=np.float32)
    cls = arr[:, 0]
    x = arr[:, 1] * w
    y = arr[:, 2] * h
    bw = arr[:, 3] * w
    bh = arr[:, 4] * h
    x1 = x - bw / 2.0
    y1 = y - bh / 2.0
    x2 = x + bw / 2.0
    y2 = y + bh / 2.0
    return np.stack([cls, x1, y1, x2, y2], axis=1)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0)


def _compute_tp_fp(
    model: YOLO,
    data_path: str,
    log_callback=None,
    verbose_logs: bool = False,
    is_onnx: bool = False,
) -> Tuple[int, int]:
    images = _collect_val_images(data_path)
    if log_callback:
        log_callback(f"Computing TP/FP on {len(images)} images...\n")

    tp = 0
    fp = 0
    total_gt = 0
    total_preds = 0
    missing_labels = 0
    redirect = _LogRedirect(log_callback) if verbose_logs else None
    
    # Fix for PyInstaller: always provide redirect to avoid None stdout/stderr
    if redirect is None:
        redirect = _LogRedirect(None)
    
    stdout_ctx = contextlib.redirect_stdout(redirect)
    stderr_ctx = contextlib.redirect_stderr(redirect)
    log_ctx = _attach_log_handler(log_callback) if verbose_logs else contextlib.nullcontext()

    predict_kwargs = {
        "source": [str(p) for p in images],
        "conf": 0.25,
        "iou": 0.5,
        "verbose": bool(verbose_logs),
    }
    if is_onnx:
        model.overrides["batch"] = 1
        predict_kwargs["batch"] = 1
        if log_callback:
            log_callback("ONNX detected: forcing batch=1 for prediction.\n")

    if is_onnx:
        results = []
        with stdout_ctx, stderr_ctx, log_ctx:
            for image_path in images:
                single_kwargs = dict(predict_kwargs)
                single_kwargs["source"] = str(image_path)
                pred = model.predict(**single_kwargs)
                if pred:
                    results.append(pred[0])
    else:
        with stdout_ctx, stderr_ctx, log_ctx:
            results = model.predict(**predict_kwargs)

    for image_path, result in zip(images, results):
        h, w = result.orig_shape
        label_path = _label_path_for_image(image_path)
        if not label_path.exists():
            missing_labels += 1
        labels = _load_labels(label_path)
        gt = _xywhn_to_xyxy(labels, w=w, h=h)
        gt_used = np.zeros((gt.shape[0],), dtype=bool)
        total_gt += gt.shape[0]

        if result.boxes is None or result.boxes.xyxy is None:
            continue

        pred_xyxy = result.boxes.xyxy.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        total_preds += pred_xyxy.shape[0]
        pred_conf = None
        if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
            pred_conf = result.boxes.conf.cpu().numpy()

        order = np.argsort(-pred_conf) if pred_conf is not None else range(len(pred_xyxy))
        for idx in order:
            pred_box = pred_xyxy[idx]
            pred_class = pred_cls[idx]

            if gt.shape[0] == 0:
                fp += 1
                continue

            class_mask = (gt[:, 0].astype(int) == pred_class) & (~gt_used)
            if not np.any(class_mask):
                fp += 1
                continue

            gt_boxes = gt[class_mask][:, 1:5]
            ious = _box_iou(pred_box, gt_boxes)
            best = np.argmax(ious) if ious.size else -1
            if best >= 0 and ious[best] >= 0.5:
                tp += 1
                gt_indices = np.where(class_mask)[0]
                gt_used[gt_indices[best]] = True
            else:
                fp += 1

    if log_callback:
        log_callback(
            "TP/FP summary: "
            f"tp={tp}, fp={fp}, images={len(images)}, gt={total_gt}, preds={total_preds}, "
            f"missing_labels={missing_labels}\n"
        )
        if total_gt > 0 and total_preds > 0 and tp == 0:
            log_callback(
                "Warning: TP=0 with GT and predictions present. "
                "This often means model/dataset class mismatch or incorrect postprocessing.\n"
            )

    return tp, fp


def _extract_metrics(results) -> Dict[str, Optional[float]]:
    box = getattr(results, "box", None)

    precision = _safe_get(box, "mp")
    if precision is None:
        precision = _safe_get(box, "p")

    recall = _safe_get(box, "mr")
    if recall is None:
        recall = _safe_get(box, "r")

    map50 = _safe_get(box, "map50")
    map50_95 = _safe_get(box, "map")

    speed_ms = None
    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        speed_ms = speed.get("inference")

    tp = None
    fp = None
    cm_obj = getattr(results, "confusion_matrix", None)
    if cm_obj is not None and hasattr(cm_obj, "matrix"):
        matrix = cm_obj.matrix
        if matrix is not None and matrix.size >= 4:
            tp_val = int(matrix[:-1, :-1].diagonal().sum())
            fp_val = int(matrix[-1, :-1].sum())
            if tp_val + fp_val > 0:
                tp = tp_val
                fp = fp_val

    if tp is None or fp is None:
        tp_sum = _sum_values(getattr(box, "tp", None))
        fp_sum = _sum_values(getattr(box, "fp", None))
        if tp is None and tp_sum is not None:
            tp = int(round(tp_sum))
        if fp is None and fp_sum is not None:
            fp = int(round(fp_sum))

    return {
        "speed_ms": speed_ms,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "tp": tp,
        "fp": fp,
    }


class _LogRedirect:
    def __init__(self, callback):
        self._callback = callback
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        self._emoji_re = re.compile(
            r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F6FF"
            r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]"
        )

    def _sanitize(self, text: str) -> str:
        cleaned = self._ansi_re.sub("", text)
        cleaned = self._emoji_re.sub("", cleaned)
        cleaned = cleaned.replace("\r", "\n")
        return cleaned

    def write(self, text):
        if not text or text is None:
            return
        try:
            cleaned = self._sanitize(text)
            if self._callback:
                self._callback(cleaned)
        except Exception:
            pass

    def flush(self):
        pass
    
    def isatty(self):
        return False


@contextlib.contextmanager
def _attach_log_handler(callback):
    handler = _CallbackLogHandler(callback)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    ultra_logger = logging.getLogger("ultralytics")
    prev_ultra_level = ultra_logger.level
    prev_ultra_propagate = ultra_logger.propagate
    root.addHandler(handler)
    ultra_logger.setLevel(logging.INFO)
    ultra_logger.propagate = True
    try:
        yield
    finally:
        root.removeHandler(handler)
        ultra_logger.setLevel(prev_ultra_level)
        ultra_logger.propagate = prev_ultra_propagate


class _CallbackLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._emoji_re = re.compile(
            r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F6FF"
            r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]"
        )

    def emit(self, record):
        if not self._callback:
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if msg:
            cleaned = self._emoji_re.sub("", msg)
            self._callback(cleaned + "\n")


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

    if log_callback:
        log_callback("Loading model...\n")

    is_onnx = model_path.lower().endswith(".onnx")

    if verbose_logs and log_callback:
        dataset = _read_dataset_yaml(data_path)
        names = dataset.get("names")
        names_count = len(names) if isinstance(names, (list, dict)) else "n/a"
        log_callback("Extra logs enabled.\n")
        log_callback(f"Python: {sys.version.split()[0]}\n")
        try:
            import ultralytics

            log_callback(f"Ultralytics: {ultralytics.__version__}\n")
        except Exception:
            log_callback("Ultralytics: unknown\n")
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log_callback(f"Torch: {torch.__version__} ({device})\n")
        except Exception:
            log_callback("Torch: unknown\n")
        log_callback(f"Model path: {model_path}\n")
        log_callback(f"Dataset yaml: {data_path}\n")
        log_callback(f"Dataset base: {dataset.get('path', 'n/a')}\n")
        log_callback(f"Dataset val: {dataset.get('val', dataset.get('test', 'n/a'))}\n")
        log_callback(f"Dataset classes: {names_count}\n")
        log_callback("Validation settings: iou=0.5\n")
        log_callback("TP/FP settings: conf=0.25, iou=0.5\n")

    redirect = _LogRedirect(log_callback) if verbose_logs else None
    
    # Fix for PyInstaller: always provide redirect to avoid None stdout/stderr
    if redirect is None:
        redirect = _LogRedirect(None)
    
    stdout_ctx = contextlib.redirect_stdout(redirect)
    stderr_ctx = contextlib.redirect_stderr(redirect)

    log_ctx = _attach_log_handler(log_callback) if verbose_logs else contextlib.nullcontext()

    with stdout_ctx, stderr_ctx, log_ctx:
        model = YOLO(model_path, task="detect") if is_onnx else YOLO(model_path)
        if is_onnx:
            model.overrides["batch"] = 1

        if log_callback:
            log_callback("Running validation...\n")

        val_kwargs = {
            "data": data_path,
            "iou": 0.5,
            "plots": False,
            "save_json": False,
            "verbose": bool(verbose_logs),
        }
        if is_onnx:
            model.overrides["batch"] = 1
            val_kwargs["batch"] = 1
            if log_callback:
                log_callback("ONNX detected: forcing batch=1 for validation.\n")

        results = model.val(**val_kwargs)

    if log_callback:
        log_callback("Collecting metrics...\n")

    metrics = _extract_metrics(results)

    if metrics.get("tp") is None or metrics.get("fp") is None:
        if log_callback:
            log_callback("TP/FP not provided by validator, running extra pass...\n")
        tp, fp = _compute_tp_fp(
            model,
            data_path,
            log_callback=log_callback,
            verbose_logs=verbose_logs,
            is_onnx=is_onnx,
        )
        metrics["tp"] = tp
        metrics["fp"] = fp

    return metrics
