#!/usr/bin/env python3
"""
Semantic-if scoring on images: "is this a cat?" in one forward pass.

Bonsai 2 27B is a vision-language model, so the same trick works on pictures.
The prompt is a lettered multiple choice with an image attached, and the answer
is read off the next-token distribution. Nothing is generated and nothing is
parsed; the option letter with the highest probability is the answer.

Worth stating plainly, because it is the one thing the local model can do that
the hosted reference cannot: **Jev is text only.** Its docs say images, audio
and video are not supported. So there is no Jev column in this table.

Differences from the text scorer:

  * the image goes through `/v1/chat/completions` with `logprobs`, not
    `/completion`, because that is the endpoint that takes an `image_url`;
  * llama-server needs `--mmproj`, the 0.63 GB vision projector, which is a
    separate download from the weights;
  * images are expensive in context. A 1.6 MB photo of a dog came to 4081
    prompt tokens on its own, which overflowed a 4096 context before the
    question was even added. The fixtures here are capped at 768px and the
    context is 8192.

    vision_score.py                       # the built-in grid
    vision_score.py -q "Is this a cat?" --image images/tiger.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import string
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "openjev"))

from score import _free_port, find_llama_server, softmax_over  # noqa: E402

LETTERS = string.ascii_uppercase
MODEL_DIR = ROOT / "bonzi" / "models" / "Ternary-Bonsai-2-27B"

SYSTEM = ("You are a decision function. Look at the image and choose the single "
          "option that best answers the question. Reply with the option letter "
          "only.")


class VisionBackend:
    """Bonsai 2 27B with its vision projector, behind llama-server."""

    def __init__(self, model_dir: Path = MODEL_DIR, gpu: bool = True,
                 ctx: int = 8192, verbose: bool = False):
        weights = next(f for f in sorted(model_dir.glob("*.gguf"))
                       if "mmproj" not in f.name.lower())
        mmproj = next((f for f in model_dir.glob("*mmproj*.gguf")), None)
        if mmproj is None:
            raise SystemExit(
                f"no mmproj in {model_dir}. The vision projector is a separate "
                f"file; download Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (0.63 GB).")
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        cmd = [find_llama_server(), "-m", str(weights), "--mmproj", str(mmproj),
               "-ngl", "99" if gpu else "0", "-c", str(ctx), "--jinja",
               "--host", "127.0.0.1", "--port", str(port)]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit("llama-server exited before becoming ready")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        props = json.loads(urllib.request.urlopen(f"{self.url}/props").read())
        if not (props.get("modalities") or {}).get("vision"):
            raise SystemExit("server reports no vision modality")

    def score_image(self, image: Path, question: str, options: list[str]) -> dict:
        b64 = base64.b64encode(Path(image).read_bytes()).decode()
        lettered = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": f"{question}\n\nOptions:\n{lettered}\n\n"
                             f"Answer with the option letter only."}]},
            ],
            "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        forward_s = time.perf_counter() - t0

        ch = out["choices"][0]
        top = ((ch.get("logprobs") or {}).get("content") or [{}])[0]
        table: dict[str, float] = {}
        for e in top.get("top_logprobs", []):
            table[e["token"]] = table.get(e["token"], 0.0) + math.exp(e["logprob"])

        logits, mass = {}, 0.0
        for i, _ in enumerate(options):
            p = table.get(LETTERS[i], 0.0)
            mass += p
            logits[LETTERS[i]] = math.log(p) if p > 0 else -1e30
        probs = {options[LETTERS.index(l)]: p
                 for l, p in softmax_over(logits).items()}
        choice = max(probs, key=probs.get) if probs else None
        return {"choice": choice, "probabilities": probs,
                "label_mass": round(mass, 6), "forward_s": round(forward_s, 3),
                "tokens": (out.get("usage") or {}).get("prompt_tokens")}

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


SUBJECTS = ["cat", "dog", "tiger", "car", "bicycle", "pizza"]

# Each image is asked about itself: the bike is asked "is this a bike?", the dog
# "is this a dog?". The wording is what a person would say, not the filename.
MATCHED = {"cat": "a cat", "dog": "a dog", "tiger": "a tiger",
           "car": "a car", "bicycle": "a bike", "pizza": "a pizza"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="image decisions in one forward pass")
    ap.add_argument("-q", "--question")
    ap.add_argument("--image")
    ap.add_argument("--options", help="comma separated; default true,false")
    ap.add_argument("--images-dir", default=str(HERE / "images"))
    ap.add_argument("--json")
    ap.add_argument("--model", default="Ternary-Bonsai-2-27B",
                    help="folder under bonzi/models; only the 27B Bonsai "
                         "models have a vision projector")
    ap.add_argument("--px", type=int,
                    help="downscale images to thislong edge before sending; "
                         "latency is roughly linear in image tokens")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    b = VisionBackend(model_dir=ROOT / "bonzi" / "models" / args.model,
                      verbose=args.verbose)
    try:
        if args.image:
            opts = ([o.strip() for o in args.options.split(",")]
                    if args.options else ["true", "false"])
            r = b.score_image(Path(args.image), args.question or "Is this a cat?", opts)
            print(json.dumps(r, ensure_ascii=False))
            return 0

        d = Path(args.images_dir)
        images = [(s, d / f"{s}.jpg") for s in SUBJECTS if (d / f"{s}.jpg").exists()]
        if args.px:
            import shutil, subprocess as sp, tempfile
            tmp = Path(tempfile.mkdtemp(prefix="bonzi-px-"))
            resized = []
            for name, src in images:
                out = tmp / src.name
                shutil.copy(src, out)
                sp.run(["sips", "-Z", str(args.px), str(out), "--out", str(out)],
                       capture_output=True)
                resized.append((name, out))
            images = resized
        if not images:
            raise SystemExit(f"no images in {d}")
        rows = []

        print('\n  each image asked about itself\n')
        print(f"  {'image':10s} {'question':22s} {'answer':7s} "
              f"{'p(true)':>9s} {'mass':>7s} {'ms':>6s}")
        for name, path in images:
            q = f"Is this {MATCHED[name]}?"
            r = b.score_image(path, q, ["true", "false"])
            ok = r["choice"] == "true"
            rows.append({"image": name, "task": "matched", "question": q,
                         **r, "correct": ok})
            print(f"  {name:10s} {q:22s} {r['choice']:7s} "
                  f"{r['probabilities']['true']:9.3f} {r['label_mass']:7.3f} "
                  f"{r['forward_s']*1000:6.0f}{'' if ok else '   <-- wrong'}")

        print(f'\n"What is the main subject?"  ({"/".join(SUBJECTS)})\n')
        print(f"  {'image':10s} {'answer':10s} {'p':>7s} {'mass':>7s} {'ms':>6s}")
        for name, path in images:
            r = b.score_image(path, "What is the main subject of this image?", SUBJECTS)
            ok = r["choice"] == name
            rows.append({"image": name, "task": "subject", **r, "correct": ok})
            print(f"  {name:10s} {r['choice']:10s} {max(r['probabilities'].values()):7.3f} "
                  f"{r['label_mass']:7.3f} {r['forward_s']*1000:6.0f}"
                  f"{'' if ok else '   <-- wrong'}")

        n = len(rows)
        print(f"\n  {sum(r['correct'] for r in rows)}/{n} correct, "
              f"median {sorted(r['forward_s'] for r in rows)[n//2]*1000:.0f} ms per image")
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=2))
    finally:
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
