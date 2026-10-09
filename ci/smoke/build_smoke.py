"""Build ci/smoke/t4-dpotrainer-smoke.ipynb (a one-off human smoke test, not a lab).

Phase 1 exit: does TRL 1.15's DPOTrainer run one step on a Colab T4 (compute capability
7.5; TRL's fused LM head uses a Triton kernel and Triton officially supports 8.0+)?
The notebook tries TRL's own path first, then the pure-PyTorch stand-in the technical
review verified, and emits an experiment record either way.

  uv run python ci/smoke/build_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

OUT = Path(__file__).with_name("t4-dpotrainer-smoke.ipynb")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"

INTRO = """# Smoke test: one DPOTrainer step on a Colab T4

**Before you run:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all.

This is not a lab. It answers one question for the workshop builders: does TRL 1.15's
`DPOTrainer` run on a T4? TRL computes log-probabilities with a Triton kernel, and Triton
officially supports GPUs of compute capability 8.0 and up; the T4 is 7.5.

The notebook tries TRL's own path, then (if that fails) a pure-PyTorch stand-in, and prints a
**run record** at the end (also offered as a download). Please send that record to the
workshop builders. It takes about 5 minutes and needs no API keys."""

SETUP = """import importlib.metadata as md, os, subprocess, sys, time
PRL_REF = "main"  #@param {type:"string"}
_keep = {"trl", "prl"}
_pins = {}  # one pin per package: the copy Python imports (Colab also has older system copies)
for _d in md.distributions():
    _n = _d.metadata["Name"]
    _k = _n.lower().replace("_", "-").replace(".", "-") if _n else None
    if _k and _k not in _pins and _k not in _keep:
        _pins[_k] = f"{_n}=={md.version(_n)}"
open("/tmp/constraints.txt", "w").write("\\n".join(sorted(_pins.values())) + "\\n")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/constraints.txt", "trl==1.15.0",
                    f"git+https://github.com/project-delphi/practical-rl@{PRL_REF}#subdirectory=prl"],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:]) if r.returncode else print("installed trl 1.15.0 and prl")
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))"""

DATA = f'''from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL, REVISION = "{MODEL}", "{REVISION}"
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
pairs = [
    {{"prompt": f"Question {{i}}: what is {{i}} plus {{i}}?", "chosen": f" The answer is {{2 * i}}.",
      "rejected": f" I am not sure, maybe {{2 * i + 1}}, or something else entirely."}}
    for i in range(8)
]
dataset = Dataset.from_list(pairs)
print(len(dataset), "preference pairs")'''

RUN = """import traceback
from trl import DPOConfig, DPOTrainer

def one_run(tag):
    policy = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32)
    args = DPOConfig(output_dir=f"/tmp/dpo-smoke-{tag}", per_device_train_batch_size=4, max_steps=2,
                     learning_rate=1e-6, beta=0.1, max_length=128, fp16=torch.cuda.is_available(),
                     bf16=False, report_to="none", save_strategy="no", logging_steps=1)
    trainer = DPOTrainer(model=policy, args=args, train_dataset=dataset, processing_class=tok)
    t0 = time.perf_counter()
    out = trainer.train()
    return {"ok": True, "loss": float(out.training_loss), "seconds": round(time.perf_counter() - t0, 1)}

results = {}
try:
    results["native"] = one_run("native")
except Exception as exc:
    results["native"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    traceback.print_exc(limit=3)
print("TRL native path:", results["native"])"""

SHIM = """if not results["native"]["ok"]:
    import torch.nn.functional as F
    import trl.trainer.utils as tu

    class TorchLogProb:  # pure-PyTorch stand-in for TRL's Triton kernel (technical review, 2026-10-09)
        @staticmethod
        def apply(h, W, b, y, T, chunk, softcap=None, scale=1.0, outputs=("log_probs",)):
            z = F.linear(h, W, b) * scale
            if softcap is not None:
                z = softcap * torch.tanh(z / softcap)
            z = z.float() / T
            logz = torch.logsumexp(z, -1)
            lp = z.gather(-1, y[:, None]).squeeze(-1) - logz
            ent = logz - (z.softmax(-1) * z).sum(-1) if "entropy" in outputs else None
            return (lp, ent, None, z.mean(-1).detach() if "mean_logits" in outputs else None,
                    (z.argmax(-1) == y) if "is_top1" in outputs else None)

    tu._ChunkedLogProbFunction = TorchLogProb
    try:
        results["shim"] = one_run("shim")
    except Exception as exc:
        results["shim"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    print("With the PyTorch stand-in:", results["shim"])
else:
    print("TRL's own path works on this GPU; no stand-in needed.")"""

RECORD = """import datetime as dt
from prl import __version__ as prl_version, record, runtime
rt = runtime.detect()
rec = {
    "schema": 1, "kind": "experiment", "experiment": "t4-dpotrainer-smoke",
    "script": "ci/smoke/t4-dpotrainer-smoke.ipynb",
    "args": {"model": MODEL, "revision": REVISION, "max_steps": 2, "batch": 4, "fp16": torch.cuda.is_available()},
    "prl_version": prl_version, "date": dt.datetime.now(dt.UTC).date().isoformat(),
    "platform": rt.platform, "env": ("colab-t4" if rt.accel == "cuda" else "colab-cpu") if rt.platform == "colab" else rt.platform,
    "hardware": {"cpu": rt.cpu, "n_cpu": rt.n_cpu, "ram_gb": rt.ram_gb, "accel": rt.accel, "gpu": rt.gpu,
                 "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None},
    "python": rt.python, "packages": runtime.package_versions(["torch", "trl", "transformers", "accelerate", "triton", "prl"]),
    "budget": {"max_steps": 2}, "seeds": [0], "per_seed": {"0": results}, "seconds": 0.0,
}
record.emit(rec)"""


def main() -> None:
    nb = nbformat.v4.new_notebook()
    cells = [
        ("intro", nbformat.v4.new_markdown_cell(INTRO)),
        ("setup", nbformat.v4.new_code_cell(SETUP)),
        ("data", nbformat.v4.new_code_cell(DATA)),
        ("run", nbformat.v4.new_code_cell(RUN)),
        ("shim", nbformat.v4.new_code_cell(SHIM)),
        ("record", nbformat.v4.new_code_cell(RECORD)),
    ]
    for cid, cell in cells:
        cell["id"] = cid
        cell.metadata["id"] = cid
    nb.cells = [c for _, c in cells]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    }
    nbformat.validate(nb)
    OUT.write_text(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
