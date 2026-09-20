#!/usr/bin/env python3
"""
One loaded model behind an HTTP API.

The point of this is that the weights load once. A Bonsai 2 27B takes tens of
seconds to load and about 6 GB of memory; running score.py per question pays
that every time. Here several different decisions -- and chat -- share one
resident model.

    POST /decide  {"state": ..., "question": ..., "options": [...]}
    POST /chat    {"message": ...}  or  {"messages": [...]}
    GET  /health

Binds to localhost and has NO AUTHENTICATION. --host 0.0.0.0 would put an
unauthenticated model server on your network; put something in front of it
first.

Usage:
    api.py --model ../bonzi/models/Ternary-Bonsai-2-27B --gpu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score import (  # noqa: E402
    DEFAULT_OPTIONS,
    LlamaCppBackend,
    TorchBackend,
    _post,
    is_gguf,
    score_one,
)

STATE = {"backend": None, "model": None, "lock": None}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quieter than the default
        sys.stderr.write(f"  {self.address_string()} {fmt % a}\n")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {
                "status": "ok",
                "model": STATE["model"],
                "backend": STATE["backend"].name,
                "endpoints": ["GET /health", "POST /decide", "POST /chat"],
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.rstrip("/")
        try:
            body = self._read_json()
            if route == "/decide":
                self._send(200, self._decide(body))
            elif route == "/chat":
                self._send(200, self._chat(body))
            else:
                self._send(404, {"error": "not found"})
        except KeyError as e:
            self._send(400, {"error": f"missing field: {e.args[0]}"})
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"bad JSON: {e}"})
        except Exception as e:  # noqa: BLE001 - report, keep serving
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _decide(self, body: dict) -> dict:
        question = body["question"]
        options = body.get("options") or list(DEFAULT_OPTIONS)
        with STATE["lock"]:
            return score_one(STATE["backend"], question, options, body.get("state"))

    def _chat(self, body: dict) -> dict:
        backend = STATE["backend"]
        if not isinstance(backend, LlamaCppBackend):
            return {"error": "/chat needs the llama.cpp backend"}
        messages = body.get("messages")
        if not messages:
            messages = [{"role": "user", "content": body["message"]}]

        think = bool(body.get("think", False))
        kwargs = {"enable_thinking": think}
        if think and body.get("effort"):
            kwargs["reasoning_effort"] = body["effort"]

        with STATE["lock"]:
            prompt = _post(
                f"{backend.url}/apply-template",
                {"messages": messages, "chat_template_kwargs": kwargs},
            )["prompt"]
            out = _post(
                f"{backend.url}/completion",
                {
                    "prompt": prompt,
                    "n_predict": int(body.get("max_tokens", 512)),
                    "temperature": float(body.get("temperature", 1.0)),
                    "top_p": float(body.get("top_p", 0.95)),
                    "top_k": int(body.get("top_k", 20)),
                },
            )
        text = out.get("content", "")
        m = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
        reasoning, reply = (m.group(1).strip(), text[m.end():]) if m else ("", text)
        return {
            "reply": reply.strip(),
            "reasoning": reasoning,
            "timings": out.get("timings", {}),
        }


def main(argv=None):
    import threading

    ap = argparse.ArgumentParser(description="one loaded model behind HTTP")
    ap.add_argument("--model", default=str(HERE.parent / "bonzi" / "models" /
                                           "Ternary-Bonsai-2-27B"))
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 exposes an UNAUTHENTICATED server; don't")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu", action="store_true", default=True,
                    help="offload to GPU (default)")
    ap.add_argument("--cpu", dest="gpu", action="store_false", help="CPU only")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--llama-server")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    model_path = Path(args.model)
    print(f"loading {model_path} ...", file=sys.stderr)
    if is_gguf(model_path):
        backend = LlamaCppBackend(
            model_path, gpu=args.gpu, ctx=args.ctx, threads=args.threads,
            llama_server=args.llama_server, verbose=args.verbose,
        )
    else:
        backend = TorchBackend(model_path, device=args.device, threads=args.threads)

    STATE.update(backend=backend, model=model_path.name, lock=threading.Lock())

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"open-bonzi-jev api on http://{args.host}:{args.port}", file=sys.stderr)
    print("  GET /health   POST /decide   POST /chat", file=sys.stderr)
    if args.host not in ("127.0.0.1", "localhost"):
        print("  WARNING: no authentication, and not bound to localhost",
              file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
    finally:
        server.server_close()
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
