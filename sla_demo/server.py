"""Local-only, dependency-light HTTP server. No outbound API or credentials."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .environment import Config
from .models import ModelSuite
from .experiment import run
from .checkpoint import load, save

WEB = Path(__file__).resolve().parent.parent / "web"


def make_handler(suite):
    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):
            # Prevent browser DNS-rebinding from treating an attacker hostname as local.
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                self.json(403, {"error": "Only localhost Host headers are accepted"})
                return False
            return True

        def send_bytes(self, status, payload, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def json(self, status, payload):
            self.send_bytes(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if not self.valid_host():
                return
            path = urlparse(self.path).path
            if path == "/api/health":
                self.json(200, {"status": "ready", "model_fingerprint": suite.fingerprint, "training": suite.report})
                return
            routes = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path not in routes:
                self.json(404, {"error": "Not found"})
                return
            file = WEB / routes[path]
            self.send_bytes(200, file.read_bytes(), (mimetypes.guess_type(str(file))[0] or "application/octet-stream") + "; charset=utf-8")

        def do_POST(self):
            if not self.valid_host():
                return
            if urlparse(self.path).path != "/api/run":
                self.json(404, {"error": "Not found"})
                return
            # Reject cross-origin browser writes; do not expose this local server publicly.
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{self.headers.get('Host')}", f"https://{self.headers.get('Host')}"}:
                self.json(403, {"error": "Cross-origin requests are not allowed"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Request must be between 1 and 8192 bytes")
                from .protocol import _unique
                data = json.loads(self.rfile.read(length), object_pairs_hook=_unique,
                                  parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
                allowed = {"config", "seed_start", "seed_count", "strength", "trace_all", "mode", "online_lr", "learning_enabled"}
                if set(data) - allowed:
                    raise ValueError("Unknown experiment fields")
                if not isinstance(data.get("config", {}), dict):
                    raise ValueError("config must be an object")
                result = run(suite, Config(**data.get("config", {})), data.get("seed_start", 100), data.get("seed_count", 24), data.get("strength", 1.0), data.get("trace_all", False), data.get("mode", "offline"), data.get("online_lr", .02), data.get("learning_enabled", True))
                self.json(200, result)
            except (TypeError, ValueError) as error:
                self.json(400, {"error": str(error)})

        def log_message(self, fmt, *args):
            return
    return Handler


def main():
    parser = argparse.ArgumentParser(description="SLA-HCI toy mechanism teaching lab")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--checkpoint", type=Path, help="Load a verified local toy NPZ instead of training")
    parser.add_argument("--save-checkpoint", type=Path, help="Save the toy suite for a future CPU run")
    args = parser.parse_args()
    if not 0 <= args.train_seed <= 2**31 - 10000:
        parser.error("train-seed must be between 0 and 2147473648")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    print("Preparing educational toy models locally. No language model or network call...", flush=True)
    try:
        suite = load(args.checkpoint) if args.checkpoint else ModelSuite(args.train_seed)
        if args.save_checkpoint:
            save(suite, args.save_checkpoint)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(suite))
    print(f"Ready: http://127.0.0.1:{args.port}  |  model {suite.fingerprint[:12]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
