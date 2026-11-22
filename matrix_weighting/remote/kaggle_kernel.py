"""Build a private Kaggle script kernel that runs a list of job files on GPU.

The kernel is one self-contained run.py. It holds the study code and the job
files as an embedded archive, unpacks them under /kaggle/working/mw, and runs
experiments/pool.py once per GPU with --device cuda. Results stay under
/kaggle/working/mw/results, so `kaggle kernels output` returns them. No
credentials are written into the kernel. Pushing uses the account
configuration given to the kaggle CLI through KAGGLE_CONFIG_DIR.

    python remote/kaggle_kernel.py --user someone --slug mw-g2-k29 --out /tmp/k1 jobs/gate2/*damped29*.json
    KAGGLE_CONFIG_DIR=... kaggle kernels push -p /tmp/k1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["mw/__init__.py", "mw/systems.py", "mw/model.py", "mw/weights.py", "mw/train.py", "mw/reference.py",
        "experiments/run_job.py", "experiments/pool.py"]

RUNNER = r'''
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

BUNDLE = "{bundle}"
WORKERS_PER_GPU = {workers}
root = Path(os.environ.get("MW_ROOT", "/kaggle/working/mw"))
root.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(root)

import torch
ngpu = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", ngpu,
      [torch.cuda.get_device_name(i) for i in range(ngpu)], flush=True)
jobs = sorted((root / "jobs" / "kaggle").glob("*.json"))
groups = max(ngpu, 1)
procs = []
for g in range(groups):
    d = root / "jobs" / f"gpu{{g}}"
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(jobs):
        if i % groups == g:
            shutil.copy(f, d / f.name)
    env = {{**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}} if ngpu else dict(os.environ)
    cmd = [sys.executable, "experiments/pool.py", str(d), "--workers", str(WORKERS_PER_GPU)]
    if ngpu:
        cmd += ["--device", "cuda"]
    procs.append(subprocess.Popen(cmd, cwd=root, env=env))
start = time.time()
codes = [p.wait() for p in procs]
print("pool exit codes", codes, "after", round(time.time() - start), "s", flush=True)
done = sorted(str(p.relative_to(root)) for p in (root / "results").rglob("summary.json"))
print(len(done), "summaries of", len(jobs), "jobs", flush=True)
for p in (root / "results").rglob("log.txt"):
    lines = p.read_text().strip().splitlines()
    print(p.parent.relative_to(root), "|", lines[-1] if lines else "", flush=True)
'''


def build(user: str, slug: str, job_files: list[Path], out: Path, workers: int) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in CODE:
            tar.add(ROOT / rel, arcname=rel)
        for f in job_files:
            tar.add(f, arcname=f"jobs/kaggle/{f.name}")
    bundle = base64.b64encode(buf.getvalue()).decode()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(RUNNER.format(bundle=bundle, workers=workers))
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3, help="concurrent jobs per GPU")
    ap.add_argument("jobs", type=Path, nargs="+")
    args = ap.parse_args()
    build(args.user, args.slug, args.jobs, args.out, args.workers)
    print(f"kernel {args.user}/{args.slug}: {len(args.jobs)} jobs -> {args.out}")


if __name__ == "__main__":
    main()
