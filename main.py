import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from evaluator import evaluate_model


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Analyzer")
        self.geometry("780x560")

        self.model_path = tk.StringVar()
        self.data_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Idle")
        self.verbose_logs = tk.BooleanVar(value=False)

        self._queue = queue.Queue()
        self._worker = None

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        path_frame = ttk.LabelFrame(main, text="Inputs", padding=10)
        path_frame.pack(fill=tk.X)

        ttk.Label(path_frame, text="Model (.pt/.onnx):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(path_frame, textvariable=self.model_path, width=64).grid(
            row=0, column=1, padx=6, sticky=tk.W
        )
        ttk.Button(path_frame, text="Browse", command=self._pick_model).grid(row=0, column=2)

        ttk.Label(path_frame, text="Dataset (data.yaml):").grid(row=1, column=0, sticky=tk.W, pady=6)
        ttk.Entry(path_frame, textvariable=self.data_path, width=64).grid(
            row=1, column=1, padx=6, sticky=tk.W
        )
        ttk.Button(path_frame, text="Browse", command=self._pick_data).grid(row=1, column=2)

        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = ttk.Button(action_frame, text="Run Evaluation", command=self._run)
        self.run_btn.pack(side=tk.LEFT)

        ttk.Checkbutton(
            action_frame,
            text="Extra logs",
            variable=self.verbose_logs,
        ).pack(side=tk.LEFT, padx=(10, 0))

        self.progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        ttk.Label(action_frame, textvariable=self.status_text).pack(side=tk.RIGHT)

        results_frame = ttk.LabelFrame(main, text="Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(12, 0))

        self._result_vars = {
            "speed_ms": tk.StringVar(value="-"),
            "precision": tk.StringVar(value="-"),
            "recall": tk.StringVar(value="-"),
            "map50": tk.StringVar(value="-"),
            "map50_95": tk.StringVar(value="-"),
            "tp": tk.StringVar(value="-"),
            "fp": tk.StringVar(value="-"),
        }

        labels = [
            ("Avg inference (ms/img)", "speed_ms"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("mAP@0.5", "map50"),
            ("mAP@0.5:0.95", "map50_95"),
            ("True positives", "tp"),
            ("False positives", "fp"),
        ]

        for idx, (label, key) in enumerate(labels):
            ttk.Label(results_frame, text=label + ":").grid(row=idx, column=0, sticky=tk.W, pady=2)
            ttk.Label(results_frame, textvariable=self._result_vars[key]).grid(
                row=idx, column=1, sticky=tk.W, pady=2
            )

        log_frame = ttk.LabelFrame(main, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(log_toolbar, text="Clear Logs", command=self._clear_logs).pack(side=tk.RIGHT)

        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _pick_model(self):
        path = filedialog.askopenfilename(
            title="Select model",
            filetypes=[("Model files", "*.pt *.onnx"), ("All files", "*.*")],
        )
        if path:
            self.model_path.set(path)

    def _pick_data(self):
        path = filedialog.askopenfilename(
            title="Select data.yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.data_path.set(path)

    def _run(self):
        model_path = self.model_path.get().strip()
        data_path = self.data_path.get().strip()

        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Input error", "Select a valid model file.")
            return
        if not data_path or not os.path.isfile(data_path):
            messagebox.showerror("Input error", "Select a valid dataset yaml file.")
            return

        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Evaluation is already running.")
            return

        self._clear_results()
        self._append_log("Starting evaluation...\n")
        self.status_text.set("Running")
        self.progress.start(10)
        self.run_btn.configure(state=tk.DISABLED)

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(model_path, data_path, self.verbose_logs.get()),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self, model_path, data_path, verbose_logs):
        try:
            results = evaluate_model(
                model_path=model_path,
                data_path=data_path,
                verbose_logs=verbose_logs,
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
                    self._apply_results(payload)
                elif kind == "error":
                    self._append_log("Error: " + payload + "\n")
                    messagebox.showerror("Evaluation failed", payload)
                    self._finish_run()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _apply_results(self, results):
        def fmt(value, digits=4):
            if value is None:
                return "-"
            if isinstance(value, int):
                return str(value)
            try:
                return f"{value:.{digits}f}"
            except Exception:
                return str(value)

        self._result_vars["speed_ms"].set(fmt(results.get("speed_ms"), digits=2))
        self._result_vars["precision"].set(fmt(results.get("precision")))
        self._result_vars["recall"].set(fmt(results.get("recall")))
        self._result_vars["map50"].set(fmt(results.get("map50")))
        self._result_vars["map50_95"].set(fmt(results.get("map50_95")))
        self._result_vars["tp"].set(fmt(results.get("tp")))
        self._result_vars["fp"].set(fmt(results.get("fp")))

        self._append_log("\nDone.\n")
        self._finish_run()

    def _finish_run(self):
        self.progress.stop()
        self.run_btn.configure(state=tk.NORMAL)
        self.status_text.set("Idle")

    def _clear_results(self):
        for var in self._result_vars.values():
            var.set("-")

    def _append_log(self, text):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _clear_logs(self):
        self.log_text.delete("1.0", tk.END)


if __name__ == "__main__":
    app = App()
    app.mainloop()
