# Ground-only Vast.ai experiments

This folder contains two comparable Qwen3.5-9B LoRA experiments for the 30 Ground classes.
Both experiments reuse the existing Ground Parquet records and existing `assistant_cot`; no new
CoT generation is required.

## Modes

* `direct`: target is changed to `<answer>Grd.X</answer>`. The model is trained to classify only.
* `answer_only`: existing `<think>...</think><answer>Grd.X</answer>` is retained, but loss is
  applied only to the answer tokens. The existing CoT is used as context and is not regenerated.

The dataset is downloaded by `datasets` from:

```text
KhanhChien/Data-Qwen3.5-SFT-Grd-Only
```

## Vast.ai setup

Recommended: H100/A100 80GB. The code also supports multi-GPU `torchrun`.

```bash
git clone https://github.com/khanhchien23/Ground-Qwen35-SFT-Vast-Experiments.git
cd Ground-Qwen35-SFT-Vast-Experiments
python -m pip install --upgrade pip
pip install -r requirements.txt
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN'
# Optional verification; login is not required by the training scripts.
huggingface-cli whoami --token "$HF_TOKEN"
```

If the folder is uploaded to a separate repository, clone that repository instead.

`HF_TOKEN` is used automatically for the dataset/model download and Hugging Face
adapter upload. Never commit the real token to GitHub.

## Train direct-label

```bash
torchrun --standalone --nproc_per_node=1 train_grd_vast.py \
  --parquet_dir KhanhChien/Data-Qwen3.5-SFT-Grd-Only \
  --output_dir ./outputs/grd_direct \
  --mode direct \
  --model_id Qwen/Qwen3.5-9B \
  --epochs 3 --batch_size 16 --grad_accum 1 \
  --lr 2e-4 --max_seq_length 512 \
  --save_steps 100 --logging_steps 10 --eval_steps 100 \
  --hub_repo KhanhChien/FineTuneQwen35-9B-Grd-Direct
```

## Train existing-CoT / answer-only-loss

```bash
torchrun --standalone --nproc_per_node=1 train_grd_vast.py \
  --parquet_dir KhanhChien/Data-Qwen3.5-SFT-Grd-Only \
  --output_dir ./outputs/grd_answer_only \
  --mode answer_only \
  --model_id Qwen/Qwen3.5-9B \
  --epochs 3 --batch_size 16 --grad_accum 1 \
  --lr 2e-4 --max_seq_length 768 \
  --save_steps 100 --logging_steps 10 --eval_steps 100 \
  --hub_repo KhanhChien/FineTuneQwen35-9B-Grd-AnswerOnly
```

For cumulative resume, use the same output directory and run `--epochs 1`, then `--epochs 2`,
then `--epochs 3`. Do not mix the two modes in one output directory.

For multiple GPUs, replace `--nproc_per_node=1` with the number of GPUs. The effective batch is
`batch_size * number_of_gpus * grad_accum`.

## Prepare real Ground evaluation data

Download `data/selected_grd_only_real.parquet` from the dataset repository. For example:

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
token = os.environ["HF_TOKEN"]
hf_hub_download(
    'KhanhChien/Data-Qwen3.5-SFT-Grd-Only',
    'data/selected_grd_only_real.parquet',
    repo_type='dataset',
    token=token,
    local_dir='./data',
)
PY
```

## Inference

Direct-label adapter:

```bash
python inference_grd_vast.py \
  --data ./data/data/selected_grd_only_real.parquet \
  --prompt-parquet ./data/data/train-00000-of-00002.parquet \
  --adapter KhanhChien/FineTuneQwen35-9B-Grd-Direct \
  --checkpoint final_adapter --mode direct \
  --batch-size 8 --max-new-tokens 32 \
  --output ./results/grd_direct.csv
```

Answer-only-loss adapter:

```bash
python inference_grd_vast.py \
  --data ./data/data/selected_grd_only_real.parquet \
  --prompt-parquet ./data/data/train-00000-of-00002.parquet \
  --adapter KhanhChien/FineTuneQwen35-9B-Grd-AnswerOnly \
  --checkpoint final_adapter --mode answer_only \
  --batch-size 8 --max-new-tokens 384 \
  --output ./results/grd_answer_only.csv
```

The CSV contains `ground_truth`, `prediction`, and `correct`. Compare exact class accuracy,
not only Ground-domain accuracy.
