"""Serve Cloudflare/clef-flash behind the System One API (`POST /v1/systemone`) via the checkpoint's own
`joint_schema_model.systemone` (model card, "Jev / SystemOne API"); the runner treats it like the other local decision
models (models/decision/local_systemone.py).

Environment: requirements/local-servers/dm.lock.txt (transformers 5.19, torchvision for the video processor).
    python scripts/serve_clef_flash.py --revision 22b75507981345858dd152fbd9a6f8c0c4068b0a --port 30000
Weights in bfloat16 (load_release_model's default); ~19 GB plus working memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Cloudflare/clef-flash")
    ap.add_argument("--revision", default="22b75507981345858dd152fbd9a6f8c0c4068b0a")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--max-length", type=int, default=16384)  # systemone()'s default
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    ckpt = snapshot_download(args.repo, revision=args.revision)
    sys.path.insert(0, ckpt)
    from joint_schema_model import load_release_model, systemone  # noqa: E402  (shipped inside the checkpoint)

    model, processor = load_release_model(ckpt, device=args.device, dtype=torch.bfloat16)
    model.eval()
    lock = Lock()  # one request at a time (batch size 1, as for every model)
    model_name = f"{args.repo}@{args.revision[:7]}"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            body.setdefault("model", args.repo)  # Clef's systemone() requires `model` (the Decider's does not)
            try:
                with lock, torch.inference_mode():
                    out = systemone(model, processor, body, max_length=args.max_length)
                out = {**out, "model": model_name}
                code = 200
            except Exception as e:  # schema / length errors are returned, not swallowed
                out, code = {"error": {"type": type(e).__name__, "message": str(e)}}, 422
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    print(f"serving {model_name} at http://127.0.0.1:{args.port}/v1/systemone", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
