"""Run Qwen3.5-9B + GRPO epoch-2 LoRA on real class-folder images or Parquet.

The GRPO adapter already contains the policy weights; load only the base model
and this adapter, not the SFT adapter a second time. Results are resumable.

Directory example:
    python inference_selected_no_errors.py --data D:\\du_lieu_that\\selected_no_errors \
      --adapter KhanhChien/GRPO-Qwen35-9B-New --checkpoint checkpoint-grpo-304-epoch2

Parquet example:
    python inference_selected_no_errors.py --data D:\\du_lieu_that\\selected_no_errors.parquet \
      --adapter D:\\models\\epoch2_adapter --output epoch2_predictions.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

sys.modules["torchao"] = None
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

# Hugging Face Hub reads this environment variable for private adapter downloads.
# The token must not be hard-coded in source files.
if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_TOKEN", os.environ["HF_TOKEN"])
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"])

MODEL_ID = "Qwen/Qwen3.5-9B"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
VALID_81_CLASSES = (
    "Air.A10", "Air.AH64", "Air.AS365", "Air.B2", "Air.B29", "Air.C130", "Air.C17",
    "Air.Casa212", "Air.CH47", "Air.CH53", "Air.F22", "Air.F35", "Air.IL76", "Air.J7",
    "Air.JH7", "Air.Ka27", "Air.Ka52", "Air.KC135", "Air.L39", "Air.Mi24", "Air.Mi28",
    "Air.Mi8", "Air.Mig29", "Air.Rafale", "Air.Su22", "Air.Su27", "Air.T6Texan",
    "Air.Yak130", "Air.Yak52", "Air.Z8", "Grd.2S1", "Grd.BM21", "Grd.BM27",
    "Grd.BM30", "Grd.BMP2", "Grd.Bradley", "Grd.BRDM2", "Grd.BTR152", "Grd.BTR60",
    "Grd.Gaz66", "Grd.Himars", "Grd.Humvee", "Grd.Kamaz", "Grd.Kraz255", "Grd.LAV25",
    "Grd.M113", "Grd.M1Abrams", "Grd.Maz537", "Grd.Merkava", "Grd.MTLB", "Grd.PT76",
    "Grd.SU100", "Grd.SU76", "Grd.T34", "Grd.T55", "Grd.T90", "Grd.Ural4320",
    "Grd.Zil131", "Grd.ZSU234", "Grd.ZSU57", "Sea.ArleighBurke", "Sea.Buyan",
    "Sea.GeraldFord", "Sea.Gorshkov", "Sea.Independence", "Sea.Kilo", "Sea.Kirov",
    "Sea.Molniya", "Sea.Nanuchka", "Sea.OsaII", "Sea.Ticonderoga",
    "Sea.Type002.Shandong", "Sea.Type022.Houbei", "Sea.Type051B.Luhai",
    "Sea.Type053H3.JiangweiII", "Sea.Type054A.JiangkaiII", "Sea.Type055.Renhai",
    "Sea.TypeAsahi", "Sea.TypeChengKung", "Sea.TypeKangDing", "Sea.Wasp",
)
SYSTEM_PROMPT = """You are an expert military analyst specializing in Fine-Grained Military Vehicle Recognition.
Your task is to inspect the image and classify the military vehicle into EXACTLY ONE of these 81 target classes:
%s
Use only the exact class spelling from this list. Use the required format:
<think>
1. Viewpoint & Environment: ...
2. Coarse Category: ...
3. Fine-Grained Visual Features: ...
4. Disambiguation & Conclusion: ...
</think>
<answer>Exact_Class_Name</answer>""" % ", ".join(VALID_81_CLASSES)
ENABLE_THINKING = True
USER_PROMPT = ("Inspect this military vehicle carefully. Provide a structured 4-step visual analysis "
                "inside <think>...</think> and conclude with the exact class label inside "
                "<answer>...</answer>.")
FIELDS = ["file_name", "ground_truth", "prediction", "correct", "generated_tokens", "seconds", "raw_output"]


def parse_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def resolve_dir(path):
    path = path.expanduser().resolve()
    if any(p.is_dir() and any(x.is_file() and x.suffix.lower() in EXTENSIONS for x in p.iterdir())
           for p in path.iterdir() if path.is_dir()):
        return path
    nested = path / path.name
    if nested.is_dir():
        return resolve_dir(nested)
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return resolve_dir(candidates[0])
    raise ValueError(f"Could not find class folders under {path}")


def items_from_data(data):
    if data.suffix.lower() == ".parquet":
        rows = load_dataset("parquet", data_files=str(data), split="train")
        return [{"file_name": r["file_name"], "ground_truth": r.get("ground_truth", ""),
                 "image_bytes": bytes(r["image_bytes"])} for r in rows]
    root = resolve_dir(data)
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS):
            rows.append({"file_name": path.relative_to(root).as_posix(),
                         "ground_truth": class_dir.name, "path": path})
    return rows


def resolve_adapter(source, checkpoint):
    local = Path(source).expanduser()
    if local.exists():
        adapter = local / checkpoint if (local / checkpoint).is_dir() else local
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter_config.json: {adapter}")
        return adapter.resolve()
    if not checkpoint:
        raise ValueError("--checkpoint is required when --adapter is a Hugging Face repo ID")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=source, allow_patterns=[f"{checkpoint}/*"],
                                  token=os.environ.get("HF_TOKEN"))) / checkpoint


def load_model(adapter):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen3.5-9B inference")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor_source = str(adapter) if (adapter / "processor_config.json").is_file() else MODEL_ID
    try:
        processor = AutoProcessor.from_pretrained(processor_source, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}, trust_remote_code=True)
    except (TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(processor_source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    # An 80GB GPU can hold Qwen3.5-9B in BF16. Avoid device_map="auto" here:
    # Accelerate may leave meta tensors, and older PEFT versions then cannot
    # inject the LoRA weights into Qwen MTP modules.
    print(f"Loading {MODEL_ID} ({dtype}) entirely on cuda:0 + adapter {adapter}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    return processor, model


def infer_batch(rows, processor, model, max_new_tokens):
    """Generate one response per image in a real padded inference batch."""
    started = time.monotonic()
    images, prompts = [], []
    try:
        for row in rows:
            if "path" in row:
                image = Image.open(row["path"]).convert("RGB")
            else:
                import io
                image = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "image", "image": image},
                                                           {"type": "text", "text": USER_PROMPT}]}]
            prompts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING))
            images.append(image)
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=processor.tokenizer.pad_token_id)
        # With left padding, every row has the same input width after padding.
        new_ids = output[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(new_ids, skip_special_tokens=False)
        elapsed = time.monotonic() - started
        return [(parse_answer(raw := "<think>\n" + text), raw, int(ids.shape[0]), elapsed)
                for ids, text in zip(new_ids, decoded)]
    finally:
        for image in images:
            image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Image root or packed .parquet")
    parser.add_argument("--adapter", required=True, help="Local adapter dir or HF repo")
    parser.add_argument("--checkpoint", default="", help="HF checkpoint subdirectory, e.g. epoch-2 checkpoint")
    parser.add_argument("--output", type=Path, default=Path("epoch2_predictions.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Images per generation batch; 32 is a starting point for an 80GB GPU")
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mode", choices=["direct", "answer_only"], default="answer_only")
    args = parser.parse_args()
    global ENABLE_THINKING, SYSTEM_PROMPT, USER_PROMPT
    ENABLE_THINKING = args.mode == "answer_only"
    if ENABLE_THINKING:
        USER_PROMPT = ("Inspect this military vehicle carefully. Provide a structured 4-step visual analysis "
                       "inside <think>...</think> and conclude with the exact class label inside <answer>...</answer>.")
    else:
        SYSTEM_PROMPT = ("Classify the Ground military vehicle in the image into exactly one of these classes: "
                         + ", ".join(c for c in VALID_81_CLASSES if c.startswith("Grd."))
                         + ". Return only <answer>Exact_Class_Name</answer>.")
        USER_PROMPT = "Inspect the image and return only the exact Ground class in <answer>...</answer>."
    if args.max_new_tokens < 1 or args.limit < 0 or args.batch_size < 1:
        parser.error("--max-new-tokens/--batch-size must be positive and --limit must be >= 0")
    rows = items_from_data(args.data.resolve())
    rows = rows[:args.limit] if args.limit else rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    jsonl_path = args.output.with_suffix(".jsonl")
    done = set()
    if not args.no_resume and args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as f:
            done = {r["file_name"] for r in csv.DictReader(f) if r.get("file_name")}
    remaining = [r for r in rows if r["file_name"] not in done]
    print(f"Dataset: {len(rows):,}; completed: {len(rows)-len(remaining):,}; remaining: {len(remaining):,}")
    if not remaining:
        return
    adapter = resolve_adapter(args.adapter, args.checkpoint) if args.checkpoint else resolve_adapter(args.adapter, "")
    processor, model = load_model(adapter)
    with args.output.open("a", newline="", encoding="utf-8") as csv_file, log_path.open("a", encoding="utf-8") as log_file, jsonl_path.open("a", encoding="utf-8") as json_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        if csv_file.tell() == 0:
            writer.writeheader()
        for offset in range(0, len(remaining), args.batch_size):
            batch = remaining[offset:offset + args.batch_size]
            try:
                results = infer_batch(batch, processor, model, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM with batch_size={len(batch)}. Re-run with --batch-size 16 or 8."
                ) from exc
            for index, (row, (prediction, raw, tokens, seconds)) in enumerate(zip(batch, results), offset + 1):
                gt = row.get("ground_truth", "")
                correct = prediction.casefold() == gt.casefold() if gt else ""
                result = {"file_name": row["file_name"], "ground_truth": gt, "prediction": prediction,
                          "correct": correct, "generated_tokens": tokens, "seconds": round(seconds, 3), "raw_output": raw}
                writer.writerow(result); csv_file.flush()
                json_file.write(json.dumps(result, ensure_ascii=False) + "\n"); json_file.flush()
                log_file.write(f"{'='*80}\nFile: {row['file_name']}\nGround truth: {gt}\nPrediction: {prediction}\nCorrect: {correct}\n{raw}\n")
                log_file.flush()
                print(f"[{index}/{len(remaining)}] {row['file_name']} -> {prediction or '(no answer)'} | gt={gt} | correct={correct}", flush=True)
    print(f"Saved: {args.output}, {log_path}, {jsonl_path}")


if __name__ == "__main__":
    main()
