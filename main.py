import os
import sys
import queue
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor

from evaluator import evaluate_model

# (display label, dict key, decimal digits)
_RESULT_FIELDS = (
    ("Avg inference (ms/img)", "speed_ms", 2),
    ("Precision",              "precision", 4),
    ("Recall",                 "recall",    4),
    ("mAP@0.5",               "map50",     4),
    ("mAP@0.5:0.95",          "map50_95",  4),
    ("True positives",         "tp",        0),
    ("False positives",        "fp",        0),
)

MONO = QFont("Consolas", 10)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Analyzer")
        self.resize(800, 600)

        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_queue)
        self._timer.start(100)

    # ── UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Inputs ---
        inp = QGroupBox("Inputs")
        inp_lay = QVBoxLayout(inp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model (.pt / .onnx):"))
        self.model_edit = QLineEdit()
        row1.addWidget(self.model_edit, 1)
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(self._pick_model)
        row1.addWidget(btn_model)
        inp_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Dataset (data.yaml):"))
        self.data_edit = QLineEdit()
        row2.addWidget(self.data_edit, 1)
        btn_data = QPushButton("Browse")
        btn_data.clicked.connect(self._pick_data)
        row2.addWidget(btn_data)
        inp_lay.addLayout(row2)

        root.addWidget(inp)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        self.run_btn = QPushButton("Run Evaluation")
        self.run_btn.clicked.connect(self._run)
        toolbar.addWidget(self.run_btn)

        self.verbose_cb = QCheckBox("Extra logs")
        toolbar.addWidget(self.verbose_cb)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # will toggle between busy / idle
        self.progress.setRange(0, 1)  # idle state: no animation
        self.progress.setValue(0)
        toolbar.addWidget(self.progress, 1)

        self.status_label = QLabel("Idle")
        toolbar.addWidget(self.status_label)

        root.addLayout(toolbar)

        # --- Results (fully selectable & copyable) ---
        res = QGroupBox("Results")
        res_lay = QVBoxLayout(res)

        res_toolbar = QHBoxLayout()
        res_toolbar.addStretch()
        btn_export = QPushButton("Export Results")
        btn_export.clicked.connect(self._export_results)
        res_toolbar.addWidget(btn_export)
        res_lay.addLayout(res_toolbar)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(MONO)
        self.results_text.setFixedHeight(len(_RESULT_FIELDS) * 22 + 10)
        res_lay.addWidget(self.results_text)

        root.addWidget(res)

        # --- Log ---
        log = QGroupBox("Log")
        log_lay = QVBoxLayout(log)

        log_toolbar = QHBoxLayout()
        log_toolbar.addStretch()
        btn_clear = QPushButton("Clear Logs")
        btn_clear.clicked.connect(self._clear_logs)
        log_toolbar.addWidget(btn_clear)
        log_lay.addLayout(log_toolbar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(MONO)
        log_lay.addWidget(self.log_text)

        root.addWidget(log, 1)

        self._clear_results()

    # ── File dialogs ────────────────────────────────────────────

    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model", "",
            "Model files (*.pt *.onnx);;All files (*)",
        )
        if path:
            self.model_edit.setText(path)

    def _pick_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "",
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.data_edit.setText(path)

    # ── Evaluation ──────────────────────────────────────────────

    def _run(self):
        model = self.model_edit.text().strip()
        data = self.data_edit.text().strip()

        if not model or not os.path.isfile(model):
            QMessageBox.critical(self, "Input error", "Select a valid model file.")
            return
        if not data or not os.path.isfile(data):
            QMessageBox.critical(self, "Input error", "Select a valid dataset yaml file.")
            return
        if self._worker and self._worker.is_alive():
            QMessageBox.information(self, "Busy", "Evaluation is already running.")
            return

        self._clear_results()
        self._append_log("Starting evaluation...\n")
        self.status_label.setText("Running")
        self.progress.setRange(0, 0)     # indeterminate / busy
        self.run_btn.setEnabled(False)

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(model, data, self.verbose_cb.isChecked()),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self, model_path: str, data_path: str, verbose: bool):
        try:
            results = evaluate_model(
                model_path=model_path,
                data_path=data_path,
                verbose_logs=verbose,
                log_callback=lambda msg: self._queue.put(("log", msg)),
            )
            self._queue.put(("results", results))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "results":
                    self._show_results(payload)
                elif kind == "error":
                    self._append_log(f"Error: {payload}\n")
                    QMessageBox.critical(self, "Evaluation failed", payload)
                    self._finish_run()
        except queue.Empty:
            pass

    def _finish_run(self):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Idle")

    # ── Results helpers ─────────────────────────────────────────

    @staticmethod
    def _fmt(value, digits: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, int) or digits == 0:
            return str(int(value))
        try:
            return f"{value:.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_results(self, results: dict) -> str:
        lines = []
        for label, key, digits in _RESULT_FIELDS:
            val = self._fmt(results.get(key), digits)
            lines.append(f"{label:<24} {val}")
        return "\n".join(lines)

    def _show_results(self, results: dict):
        self.results_text.setPlainText(self._format_results(results))
        self._append_log("\nDone.\n")
        self._finish_run()

    def _clear_results(self):
        self.results_text.setPlainText(self._format_results({}))

    def _export_results(self):
        text = self.results_text.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)

    # ── Log helpers ─────────────────────────────────────────────

    def _append_log(self, text: str):
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def _clear_logs(self):
        self.log_text.clear()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
