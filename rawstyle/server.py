"""Local browser interface for RAWStyle Studio."""
from __future__ import annotations

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import threading
import traceback
from urllib.parse import urlparse
import webbrowser

from .catalog import read_catalog, pair_raws
from .export import create_predictions_file
from .pipeline import enumerate_raw_files, analyse_raws
from .model import MIN_MATCHED, open_model, train, predict, save_model

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME = (Path.home() / "Library" / "Application Support" / "RAWStyle Studio") if platform.system() == "Darwin" else (Path.home() / ".rawstyle-studio")
MODEL_FILE = DATA_HOME / "my-style.joblib"
CONFIG_FILE = DATA_HOME / "preferences.json"
STATIC = APP_ROOT / "rawstyle" / "index.html"


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.session_token = secrets.token_urlsafe(24)
        self.state = {
            "catalog": "", "training_folder": "", "new_shoot": "",
            "busy": False, "mode": "idle", "done": 0, "total": 0,
            "message": "Ready.", "error": "",
            "trained_count": None, "matched_count": None, "validation": {},
            "output": "", "predicted_count": 0, "samples": [],
        }
        if CONFIG_FILE.is_file():
            try:
                prior = json.loads(CONFIG_FILE.read_text(encoding="utf8"))
                for key in ("catalog", "training_folder", "new_shoot"):
                    if isinstance(prior.get(key), str):
                        self.state[key] = prior[key]
            except (OSError, ValueError):
                pass

    def snapshot(self):
        with self.lock:
            return dict(self.state, model_ready=MODEL_FILE.is_file())

    def set(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def config_save(self):
        DATA_HOME.mkdir(parents=True, exist_ok=True)
        state = self.snapshot()
        text = json.dumps({k: state[k] for k in ("catalog", "training_folder", "new_shoot")}, indent=2)
        CONFIG_FILE.write_text(text, encoding="utf8")

    def choose(self, kind):
        if kind not in ("catalog", "training_folder", "new_shoot"):
            raise ValueError("Unknown file-picker target")
        if platform.system() != "Darwin":
            raise ValueError("Native Finder selection is available on Mac. Use the path input below on other systems.")
        if kind == "catalog":
            apple_script = 'POSIX path of (choose file with prompt "Select your Lightroom Classic .lrcat catalog")'
        elif kind == "training_folder":
            apple_script = 'POSIX path of (choose folder with prompt "Select the folder containing your edited RAW originals")'
        else:
            apple_script = 'POSIX path of (choose folder with prompt "Select your new RAW shoot")'
        proc = subprocess.run(["/usr/bin/osascript", "-e", apple_script],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode:
            if "User canceled" in proc.stderr or "-128" in proc.stderr:
                return {"cancelled": True}
            raise ValueError(proc.stderr.strip() or "Could not open Finder selection.")
        return self.select_path(kind, proc.stdout.strip())

    def select_path(self, kind, value):
        if kind not in ("catalog", "training_folder", "new_shoot"):
            raise ValueError("Unknown selection")
        path = Path(value).expanduser().resolve()
        if kind == "catalog":
            if not path.is_file() or path.suffix.lower() != ".lrcat":
                raise ValueError("Choose a .lrcat Lightroom Classic catalog file.")
        elif not path.is_dir():
            raise ValueError("Choose an existing RAW image folder.")
        self.set(**{kind: str(path)}, message=f"Selected {path.name}", error="")
        self.config_save()
        return {"path": str(path)}

    def start(self, mode, workers, ram_mb):
        if mode not in ("train", "predict"):
            raise ValueError("Unknown action")
        workers = max(1, min(6, int(workers)))
        ram_mb = max(64, min(2048, int(ram_mb)))
        with self.lock:
            if self.state["busy"]:
                raise ValueError("An operation is already running.")
            self.state.update(busy=True, mode=mode, done=0, total=0, error="", output="" if mode == "predict" else self.state["output"],
                              message="Preparing your RAW images…")
        task = self._train if mode == "train" else self._predict
        thread = threading.Thread(target=self._run, args=(task, workers, ram_mb), daemon=True)
        thread.start()
        return {"started": True}

    def _run(self, task, workers, ram_mb):
        try:
            import rawpy  # noqa: F401 - fail immediately if not installed
            task(workers, ram_mb)
        except Exception as exc:
            traceback.print_exc()
            self.set(error=str(exc), message="Could not finish: " + str(exc))
        finally:
            self.set(busy=False, mode="idle")

    def _progress(self, index, total, filename):
        self.set(done=index, total=total,
                 message=f"Reading sensor data · {index:,} of {total:,} · {filename}")

    def _train(self, workers, ram_mb):
        state = self.snapshot()
        if not state["catalog"] or not state["training_folder"]:
            raise ValueError("Choose a Lightroom catalog and the folder with its original RAW images.")
        self.set(message="Matching Lightroom adjustments to RAW files…")
        records = read_catalog(state["catalog"])
        files = enumerate_raw_files(state["training_folder"])
        paths, labels, report = pair_raws(records, files)
        self.set(matched_count=report["matched"])
        if report["matched"] < MIN_MATCHED:
            raise ValueError(
                f"Matched {report['matched']} of {report['catalog_photos']} Lightroom photographs "
                f"to {report['raw_photos']} RAW files. At least {MIN_MATCHED} matched pairs are required. "
                "Ambiguous duplicate filenames are skipped."
            )
        self.set(total=len(paths), message="Loading original RAW sensor data…")
        features = analyse_raws(paths, progress=self._progress, workers=workers, ram_mb=ram_mb)
        successes = [(v, s) for v, s in zip(features, labels) if isinstance(v, list)]
        failures = [(str(path), err) for path, err in zip(paths, features) if not isinstance(err, list)]
        if len(successes) < MIN_MATCHED:
            raise ValueError(
                f"Only {len(successes)} RAW files could be decoded; need {MIN_MATCHED}. " +
                ("First issue: " + failures[0][1] if failures else "")
            )
        self.set(message=f"Training a model from {len(successes)} RAW/edit pairs…")
        vectors, settings = map(list, zip(*successes))
        model = train(vectors, settings)
        save_model(model, MODEL_FILE)
        self.set(trained_count=len(successes), validation=model["validation_mae"],
                 message=f"Model trained on {len(successes)} RAW/edit pairs.",
                 error="")

    def _predict(self, workers, ram_mb):
        state = self.snapshot()
        if not state["new_shoot"]:
            raise ValueError("Choose the folder containing your new RAW shoot.")
        model = open_model(MODEL_FILE)
        if model is None:
            raise ValueError("Train a model before predicting adjustments.")
        paths = enumerate_raw_files(state["new_shoot"])
        if not paths:
            raise ValueError("No supported RAW images found in that folder.")
        self.set(total=len(paths), message="Analysing the new shoot from RAW sensor data…")
        features = analyse_raws(paths, progress=self._progress, workers=workers, ram_mb=ram_mb)
        good = [(p, v) for p, v in zip(paths, features) if isinstance(v, list)]
        bad = [(p, v) for p, v in zip(paths, features) if not isinstance(v, list)]
        if not good:
            raise ValueError("No RAW files could be decoded: " + str(bad[0][1]))
        self.set(message="Predicting Lightroom sliders, including Kelvin…")
        output = predict(model, [v for _, v in good])
        pairs = [(p, s) for (p, _), s in zip(good, output)]
        filename = create_predictions_file(pairs, DATA_HOME / "Predictions")
        sample = [{"file": p.name,
                   "temp": s.get("Temperature"),
                   "tint": s.get("Tint"),
                   "exposure": s.get("Exposure2012")}
                  for p, s in pairs[:6]]
        self.set(output=str(filename), predicted_count=len(pairs), samples=sample, error="",
                 message=f"Predicted {len(pairs)} RAW files." + (f" {len(bad)} could not be decoded." if bad else ""))

    def reveal_predictions(self):
        filename = self.snapshot()["output"]
        if not filename or not Path(filename).is_file():
            raise ValueError("No predictions file yet. Predict your new shoot first.")
        if platform.system() == "Darwin":
            subprocess.Popen(["/usr/bin/open", "-R", filename])
        else:
            return {"path": filename}
        return {"path": filename}


class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, fmt, *args):
        # Avoid noisy request logs during progress polling.
        return

    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            html = STATIC.read_bytes()
            html = html.replace(b"__RAWSTYLE_SESSION__", self.app.session_token.encode("ascii"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif route == "/api/status":
            self._json(self.app.snapshot())
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            if self.headers.get("X-RAWStyle-Session") != self.app.session_token:
                return self._json({"error": "Invalid local browser session."}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json({"error": "Expected JSON input."}, 415)
            size = int(self.headers.get("Content-Length", "0"))
            if size > 16 * 1024 or size < 0:
                return self._json({"error": "Request too large."}, 413)
            data = json.loads(self.rfile.read(size) or b"{}")
            route = urlparse(self.path).path
            if route == "/api/choose":
                reply = self.app.choose(data.get("kind"))
            elif route == "/api/select":
                reply = self.app.select_path(data.get("kind"), data.get("path", ""))
            elif route == "/api/run":
                reply = self.app.start(data.get("mode"), data.get("workers", 3), data.get("ram_mb", 384))
            elif route == "/api/reveal":
                reply = self.app.reveal_predictions()
            else:
                return self._json({"error": "Unknown command."}, 404)
            self._json(reply)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def main():
    import socket
    app = App()
    class BoundHandler(Handler):
        pass
    BoundHandler.app = app
    http = ThreadingHTTPServer(("127.0.0.1", 0), BoundHandler)
    port = http.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print("RAWStyle Studio is running at", url, flush=True)
    print("Close this Terminal window to stop the local server.", flush=True)
    webbrowser.open(url)
    try:
        http.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        print("Stopping RAWStyle Studio.")
    finally:
        http.server_close()


if __name__ == "__main__":
    main()
