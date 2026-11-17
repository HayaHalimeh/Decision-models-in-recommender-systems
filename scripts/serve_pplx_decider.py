"""Serve perplexity-ai/pplx-decider-flash-v1.2 behind the System One API (`POST /v1/systemone`), so the runner treats it
exactly like Jev / MS-Decision-1 / Clef / Kev-4B (models/decision/local_systemone.py).

Environment: requirements/local-servers/pplx.lock.txt (Python 3.12, the checkpoint's requirements; model card "Usage").
    python scripts/serve_pplx_decider.py --revision 9ff7cf9838e59df461c276b7699d56b244d2af58 --port 8010
The checkpoint's Decider loads its weights in bfloat16 on a GPU (it takes no dtype argument); the reference path
(`fast=False`, its default) is used. Needs a GPU with room for ~9 GiB of weights plus working memory (model card).
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="perplexity-ai/pplx-decider-flash-v1.2")
    ap.add_argument("--revision", default="9ff7cf9838e59df461c276b7699d56b244d2af58")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--max-length", type=int, default=32768)  # model card: stay within the training range
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    ckpt = Path(snapshot_download(args.repo, revision=args.revision))
    sys.path.insert(0, str(ckpt / "source" / "src"))
    from pplx_decider import Decider  # noqa: E402  (shipped inside the checkpoint)

    decider = Decider.from_pretrained(ckpt, device=args.device, fast=False, max_length=args.max_length)
    dtype = next(decider.backbone.parameters()).dtype
    lock = Lock()  # one request at a time (batch size 1, as for every model)
    model_name = f"{args.repo}@{args.revision[:7]}"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            body.pop("model", None)
            try:
                with lock:
                    out = decider.decide(body)
                out = {"model": model_name, **out}
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

    print(f"serving {model_name} ({dtype}) at http://127.0.0.1:{args.port}/v1/systemone", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
