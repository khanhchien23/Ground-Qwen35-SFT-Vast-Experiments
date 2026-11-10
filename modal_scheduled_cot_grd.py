#!/usr/bin/env python3
"""Run Ground Scheduled-CoT SFT on a persistent Modal H100.

Examples:
  modal run --detach modal_scheduled_cot_grd.py::train \
    --hf-token hf_xxx --stop-after-epoch 1

Keep ``epochs=3`` fixed for every resume.  Only change stop_after_epoch to
pause after epoch 1, 2, and 3.  The trainer's checkpoint/scheduler state stays
in the Modal Volume; the epoch adapter is also copied to the Hub.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
DATASET_REPO = "KhanhChien/Data-Qwen3.5-SFT-Grd-Only"
MODEL_ID = "Qwen/Qwen3.5-9B"
HUB_REPO = "KhanhChien/FineTuneQwen35-9B-Grd-ScheduledCoT"
VOLUME_NAME = "qwen35-grd-scheduled-cot-volume"
OUTPUT = "/data/outputs/grd_scheduled_cot"

app = modal.App("qwen35-grd-scheduled-cot")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# causal-conv1d prevents Qwen3.5 from falling back to the much slower pure
# PyTorch convolution path.  flash-linear-attention is retained for Qwen MTP.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "build-essential")
    .pip_install("torch>=2.6.0", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers>=4.57.0", "accelerate>=1.2.0", "peft>=0.14.0",
        "qwen-vl-utils", "datasets", "pyarrow", "huggingface_hub>=0.24",
        "pillow", "tqdm", "tensorboard", "wandb", "flash-linear-attention",
        "causal-conv1d>=1.5.0",
    )
    .add_local_file(ROOT / "train_grd_vast.py", "/root/train_grd_vast.py")
    .add_local_file(ROOT / "scheduled_cot.py", "/root/scheduled_cot.py")
    .add_local_file(ROOT / "ground_prompts.py", "/root/ground_prompts.py")
)


def authenticate(token: str):
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={"/data": volume},
    cpu=16.0,
    memory=65536,
)
def train_remote(
    hf_token: str,
    dataset_repo: str,
    hub_repo: str,
    epochs: int,
    stop_after_epoch: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    save_steps: int,
    max_seq_length: int,
    model_id: str,
    logging_steps: int,
    eval_steps: int,
    lora_r: int,
    lora_alpha: int,
):
    authenticate(hf_token)
    from huggingface_hub import HfApi, snapshot_download

    dataset_dir = snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        token=hf_token,
        local_dir="/data/datasets/" + dataset_repo.replace("/", "__"),
        allow_patterns=["data/train*.parquet", "data/validation*.parquet"],
    )

    cmd = [
        sys.executable, "/root/train_grd_vast.py",
        "--model_id", model_id,
        "--parquet_dir", dataset_dir,
        "--output_dir", OUTPUT,
        "--mode", "scheduled_cot",
        "--hub_repo", hub_repo,
        "--epochs", str(epochs),
        "--stop_after_epoch", str(stop_after_epoch),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--lr", str(lr),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--logging_steps", str(logging_steps),
        "--eval_steps", str(eval_steps),
        "--save_steps", str(save_steps),
        "--max_seq_length", str(max_seq_length),
        "--cot_weight_start", "1.0",
        "--cot_weight_end", "0.5",
        "--answer_weight", "1.0",
        "--epoch_end_mode",
    ]
    print("--> Modal command:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd="/root")

        # Keep a named immutable adapter for inference even when the next
        # cumulative run later replaces final_adapter on the volume.
        epoch_dir = Path(OUTPUT) / f"epoch-{stop_after_epoch}"
        shutil.copytree(Path(OUTPUT) / "final_adapter", epoch_dir, dirs_exist_ok=True)
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=hub_repo,
            folder_path=str(epoch_dir),
            path_in_repo=f"epoch-{stop_after_epoch}",
            commit_message=f"Save Scheduled CoT epoch {stop_after_epoch} adapter",
        )
        print(f"--> Uploaded {hub_repo}/epoch-{stop_after_epoch}", flush=True)
    finally:
        volume.commit()


def token_or_error(token: str) -> str:
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Pass --hf-token hf_... or set HF_TOKEN")
    return token


@app.local_entrypoint()
def train(
    hf_token: str = "",
    dataset_repo: str = DATASET_REPO,
    hub_repo: str = HUB_REPO,
    epochs: int = 3,
    stop_after_epoch: int = 1,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 2e-4,
    save_steps: int = 100,
    max_seq_length: int = 2048,
    model_id: str = MODEL_ID,
    logging_steps: int = 10,
    eval_steps: int = 100,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    if epochs != 3:
        raise ValueError("Scheduled CoT requires epochs=3; keep the total schedule fixed")
    if not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must be 1, 2, or 3")
    if min(batch_size, grad_accum, save_steps, max_seq_length, logging_steps, eval_steps) < 1:
        raise ValueError("Training parameters must be positive")
    train_remote.remote(
        token_or_error(hf_token), dataset_repo, hub_repo, epochs, stop_after_epoch,
        batch_size, grad_accum, lr, save_steps, max_seq_length, model_id,
        logging_steps, eval_steps, lora_r, lora_alpha,
    )
