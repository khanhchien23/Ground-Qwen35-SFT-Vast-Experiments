# Ground-only Vast.ai experiments

This folder contains Qwen3.5-9B LoRA experiments for the 30 Ground classes.
All modes reuse the existing Ground Parquet records and existing `assistant_cot`; no new
CoT generation is required.

## Modes

* `direct`: target is changed to `<answer>Grd.X</answer>`. The model is trained to classify only.
* `answer_only`: existing `<think>...</think><answer>Grd.X</answer>` is retained, but loss is
  applied only to the answer tokens. The existing CoT is used as context and is not regenerated.
* `scheduled_cot`: mean CE separately per reasoning/answer segment per example, then
  `loss = cosine_weight * mean_cot + 1.0 * mean_answer`, averaged across examples.
  The CoT weight goes from 1.0 to 0.5 over the **whole planned run**, using optimizer
  `global_step / max_steps`. `</think>`, the answer and turn termination belong to
  the answer segment. Prompt, image and padding tokens never receive loss.

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
  --epochs 3 --batch_size 2 --grad_accum 8 \
  --lr 2e-4 --max_seq_length 1280 \
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
  --epochs 3 --batch_size 2 --grad_accum 8 \
  --lr 2e-4 --max_seq_length 2048 \
  --save_steps 100 --logging_steps 10 --eval_steps 100 \
  --hub_repo KhanhChien/FineTuneQwen35-9B-Grd-AnswerOnly
```

Keep `--epochs 3` fixed from the beginning. To pause for inference, use
`--stop_after_epoch 1`, then `2`, then `3` with the same output directory.
Changing total epochs on resume changes the cosine LR/CoT schedules.
Do not mix modes in one output directory or Hub repository.

## Scheduled CoT: train epoch 1 of a fixed three-epoch plan

Reuse the Parquets already downloaded yesterday in `./.hf_dataset_files/data`:

```bash
torchrun --standalone --nproc_per_node=1 train_grd_vast.py \
  --parquet_dir ./.hf_dataset_files/data \
  --output_dir ./outputs/grd_scheduled_cot \
  --mode scheduled_cot --model_id Qwen/Qwen3.5-9B \
  --epochs 3 --stop_after_epoch 1 \
  --batch_size 2 --grad_accum 8 --lr 2e-4 \
  --lora_r 16 --lora_alpha 32 --max_seq_length 2048 \
  --cot_weight_start 1.0 --cot_weight_end 0.5 --answer_weight 1.0 \
  --save_steps 100 --logging_steps 10 --eval_steps 100 \
  --hub_repo KhanhChien/FineTuneQwen35-9B-Grd-ScheduledCoT
```

If the local cache is elsewhere, use its actual directory, or pass the existing
dataset ID `KhanhChien/Data-Qwen3.5-SFT-Grd-Only` instead. No CoT is regenerated.
2048 is an initial maximum, not truncation: increase if the full sequence exceeds it.
Longer CoT may need more VRAM than Direct. If batch 2 OOMs, use batch 1 / grad_accum 16
**before starting the experiment** and keep those settings unchanged on resume.

For epoch 2 and epoch 3, repeat the exact command, changing **only**
`--stop_after_epoch 2` or `--stop_after_epoch 3`. All schedules remain based on
three epochs. This mode automatically saves/evaluates at epoch boundaries and
disables loading best-validation weights at the end: `final_adapter` is the
actual current epoch-end model. The full `checkpoint-*` retains optimizer,
scheduler, RNG and Trainer state for resume. `scheduled_cot_config.json` inside
each checkpoint rejects incompatible resume schedules. Do not delete checkpoints
or resume from `final_adapter`, which contains inference weights only.

Weight at epoch boundaries: start 1.0; epoch 1 about 0.875; epoch 2 about 0.625;
epoch 3 0.5. TensorBoard logs include `cot_weight` and `answer_weight`.
Validation loss uses the current scheduled weight, so values across time are not
a fixed objective and should not be treated as classification accuracy.

Inference after the pause (change output name for each epoch):

```bash
python inference_grd_vast.py \
  --data ./data/data/selected_grd_only_real.parquet \
  --prompt-parquet ./.hf_dataset_files/data/train-grd-only-00000-of-00002.parquet \
  --adapter ./outputs/grd_scheduled_cot/final_adapter \
  --mode scheduled_cot --batch-size 1 --max-new-tokens 512 \
  --output ./results/grd_scheduled_cot_epoch1.csv
```

Unlike Direct, this mode generates reasoning at inference and uses the original
training prompts. Check whether 512 tokens suffices to reach `</answer>`; do not
silently treat truncated reasoning as a valid classification output.

## Modal inference using the immutable epoch-1 adapter

After uploading the code with `git pull`, run from Windows CMD:

```cmd
modal run --detach modal_scheduled_cot_grd.py::infer ^
  --hf-token %HF_TOKEN% ^
  --checkpoint epoch-1 ^
  --batch-size 1 ^
  --max-new-tokens 512 ^
  --output-name grd_scheduled_cot_epoch1.csv
```

The command downloads `selected_grd_only_real.parquet`, filters the 681 Ground
images, loads `epoch-1` from `KhanhChien/FineTuneQwen35-9B-Grd-ScheduledCoT`,
and writes CSV/JSONL/log files to the persistent Modal Volume under
`/data/inference_outputs/`. It uses `--no-resume` so the named result is rebuilt
from scratch, without training or changing any checkpoint.

CPU tests (no model download): `python -m unittest test_scheduled_cot -v`.
These verify masking, causal shift, padding, segment length independence and
loss gradients; an end-to-end Vast.ai GPU run is still required.

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
  --adapter KhanhChien/FineTuneQwen35-9B-Grd-Direct \
  --checkpoint final_adapter --mode direct \
  --batch-size 1 --max-new-tokens 32 \
  --output ./results/grd_direct.csv
```

Answer-only-loss adapter:

```bash
python inference_grd_vast.py \
  --data ./data/data/selected_grd_only_real.parquet \
  --prompt-parquet ./.hf_dataset_files/data/train-grd-only-00000-of-00002.parquet \
  --adapter KhanhChien/FineTuneQwen35-9B-Grd-AnswerOnly \
  --checkpoint final_adapter --mode answer_only \
  --batch-size 1 --max-new-tokens 384 \
  --output ./results/grd_answer_only.csv
```

The CSV contains `ground_truth`, `prediction`, and `correct`. Compare exact class accuracy,
not only Ground-domain accuracy.
## Hierarchical 81-class router -> domain specialists

`hierarchical_inference.py` implements the complete two-stage test flow:

1. Load the 81-class router LoRA and predict a class from the image.
2. Derive `Air`, `Ground`, or `Sea` from the router prediction prefix.
3. Load Air/Ground/Sea LoRAs on the same Qwen3.5-9B base.
4. Group images by predicted domain and run the matching specialist adapter.
5. Save router output, selected adapter, specialist output, domain correctness,
   final correctness, raw responses, and timings to CSV.

All four adapters must use the same `Qwen/Qwen3.5-9B` base and compatible PEFT
configuration. For Ground Scheduled CoT, pass the original training prompt
Parquet with `--ground-prompt-parquet`; do not replace it with a Direct prompt.

Example on Vast (Bash; replace the placeholder adapters):

```bash
python hierarchical_inference.py \
  --data ./data/realtest.parquet \
  --router-adapter ROUTER_ADAPTER_OR_REPO \
  --router-checkpoint ROUTER_CHECKPOINT_IF_HF \
  --air-adapter AIR_ADAPTER_OR_REPO \
  --air-checkpoint AIR_CHECKPOINT_IF_HF \
  --ground-adapter ./outputs/grd_scheduled_cot/final_adapter \
  --sea-adapter SEA_ADAPTER_OR_REPO \
  --sea-checkpoint SEA_CHECKPOINT_IF_HF \
  --ground-prompt-parquet ./.hf_dataset_files/data/train-grd-only-00000-of-00002.parquet \
  --output ./results/hierarchical_test.csv \
  --batch-size 1 --limit 10 --no-resume
```

Use `--limit 10` first to validate adapter loading and output format. Remove
the limit only after that test succeeds. The router never reads ground truth to
select an adapter; ground truth is used only for evaluation columns.
