#!/usr/bin/env python
"""Two-stage 81-class router -> domain LoRA specialist inference.

The script deliberately keeps routing independent from the ground-truth label:

    image -> router adapter (81 classes) -> Air/Ground/Sea -> specialist adapter

All adapters must be PEFT adapters trained from the same base model. The base
model is loaded once and adapters are switched in-place. For best throughput,
the script routes the complete dataset first, then runs each domain group in a
single adapter pass.

Examples (Bash):
    python hierarchical_inference.py \
      --data ./data/real.parquet \
      --router-adapter KhanhChien/ROUTER_REPO \
      --router-checkpoint final_adapter \
      --air-adapter ./adapters/air \
      --ground-adapter ./outputs/grd_scheduled_cot/final_adapter \
      --sea-adapter ./adapters/sea \
      --ground-prompt-parquet ./.hf_dataset_files/data/train-grd-only-00000-of-00002.parquet \
      --output ./results/hierarchical.csv --batch-size 1 --no-resume

For a Hugging Face adapter repository, ``--*-checkpoint`` is the directory
inside that repository. For a local adapter directory, the checkpoint option
is normally omitted.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.modules["torchao"] = None
import torch
import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

import inference_selected_no_errors as engine


MODEL_ID = "Qwen/Qwen3.5-9B"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

AIR_CLASSES = (
    "Air.A10", "Air.AH64", "Air.AS365", "Air.B2", "Air.B29", "Air.C130",
    "Air.C17", "Air.Casa212", "Air.CH47", "Air.CH53", "Air.F22", "Air.F35",
    "Air.IL76", "Air.J7", "Air.JH7", "Air.Ka27", "Air.Ka52", "Air.KC135",
    "Air.L39", "Air.Mi24", "Air.Mi28", "Air.Mi8", "Air.Mig29", "Air.Rafale",
    "Air.Su22", "Air.Su27", "Air.T6Texan", "Air.Yak130", "Air.Yak52", "Air.Z8",
)
GROUND_CLASSES = (
    "Grd.2S1", "Grd.BM21", "Grd.BM27", "Grd.BM30", "Grd.BMP2", "Grd.Bradley",
    "Grd.BRDM2", "Grd.BTR152", "Grd.BTR60", "Grd.Gaz66", "Grd.Himars",
    "Grd.Humvee", "Grd.Kamaz", "Grd.Kraz255", "Grd.LAV25", "Grd.M113",
    "Grd.M1Abrams", "Grd.Maz537", "Grd.Merkava", "Grd.MTLB", "Grd.PT76",
    "Grd.SU100", "Grd.SU76", "Grd.T34", "Grd.T55", "Grd.T90", "Grd.Ural4320",
    "Grd.Zil131", "Grd.ZSU234", "Grd.ZSU57",
)
SEA_CLASSES = (
    "Sea.ArleighBurke", "Sea.Buyan", "Sea.GeraldFord", "Sea.Gorshkov",
    "Sea.Independence", "Sea.Kilo", "Sea.Kirov", "Sea.Molniya", "Sea.Nanuchka",
    "Sea.OsaII", "Sea.Ticonderoga", "Sea.Type002.Shandong", "Sea.Type022.Houbei",
    "Sea.Type051B.Luhai", "Sea.Type053H3.JiangweiII", "Sea.Type054A.JiangkaiII",
    "Sea.Type055.Renhai", "Sea.TypeAsahi", "Sea.TypeChengKung",
    "Sea.TypeKangDing", "Sea.Wasp",
)
ALL_CLASSES = AIR_CLASSES + GROUND_CLASSES + SEA_CLASSES

DOMAIN_CLASSES = {"Air": AIR_CLASSES, "Ground": GROUND_CLASSES, "Sea": SEA_CLASSES}
DOMAIN_PREFIX = {"Air": "Air.", "Ground": "Grd.", "Sea": "Sea."}
DOMAIN_ADAPTER = {"Air": "air", "Ground": "ground", "Sea": "sea"}

ROUTER_SYSTEM = (
    "You are an expert military image classifier. Classify the image into "
    "exactly one of these 81 classes: " + ", ".join(ALL_CLASSES) + ". "
    "Return exactly <answer>Exact_Class_Name</answer>."
)
ROUTER_USER = "Inspect the image and return the exact class inside <answer>...</answer>."

SPECIALIST_SYSTEM = {
    domain: (
        f"Classify the {domain} military vehicle in the image into exactly one "
        f"of these classes: {', '.join(classes)}. "
        "Return exactly <answer>Exact_Class_Name</answer>."
    ) for domain, classes in DOMAIN_CLASSES.items()
}
SPECIALIST_USER = {
    domain: f"Inspect this image and return exactly one {domain} class in <answer>...</answer>."
    for domain in DOMAIN_CLASSES
}


def parse_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def resolve_adapter(source, checkpoint=""):
    local = Path(source).expanduser()
    if local.exists():
        adapter = local / checkpoint if checkpoint and (local / checkpoint).is_dir() else local
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter_config.json: {adapter}")
        return str(adapter.resolve())
    if not checkpoint:
        raise ValueError(f"--checkpoint is required for Hugging Face adapter repo: {source}")
    from huggingface_hub import snapshot_download
    return str(Path(snapshot_download(
        repo_id=source, allow_patterns=[f"{checkpoint}/*"], token=os.environ.get("HF_TOKEN")
    )) / checkpoint)


def load_prompt(parquet_path):
    if not parquet_path:
        return None
    table = pq.read_table(Path(parquet_path), columns=["system_prompt", "user_prompt"])
    if not len(table):
        raise ValueError(f"Prompt Parquet is empty: {parquet_path}")
    return table["system_prompt"][0].as_py(), table["user_prompt"][0].as_py()


def load_items(data):
    data = Path(data).expanduser().resolve()
    if data.is_file():
        columns = set(pq.read_schema(data).names)
        gt_col = "ground_truth" if "ground_truth" in columns else "gt_class" if "gt_class" in columns else None
        wanted = [x for x in ("file_name", gt_col, "image_bytes") if x]
        rows = pq.read_table(data, columns=wanted).to_pylist()
        return [{"file_name": r.get("file_name", str(i)),
                 "ground_truth": r.get(gt_col, "") if gt_col else "",
                 "image_bytes": bytes(r["image_bytes"])} for i, r in enumerate(rows)]
    if not data.is_dir():
        raise FileNotFoundError(data)
    folders = [p for p in data.iterdir() if p.is_dir()]
    rows = []
    for folder in sorted(folders):
        for image in sorted(folder.rglob("*")):
            if image.is_file() and image.suffix.lower() in EXTENSIONS:
                rows.append({"file_name": image.relative_to(data).as_posix(),
                             "ground_truth": folder.name, "path": image})
    if not rows:
        raise ValueError(f"No images found under {data}")
    return rows


def image_for(row):
    if "path" in row:
        return Image.open(row["path"]).convert("RGB")
    return Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")


def load_model(router_path, processor_source=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        processor = AutoProcessor.from_pretrained(
            processor_source or MODEL_ID,
            size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            trust_remote_code=True,
        )
    except (TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(
            processor_source or MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
            trust_remote_code=True,
        )
    processor.tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, router_path, adapter_name="router", is_trainable=False).eval()
    return processor, model


def add_adapters(model, paths):
    for name, path in paths.items():
        print(f"Loading adapter {name}: {path}", flush=True)
        model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.eval()


def run_batch(rows, processor, model, adapter_name, system_prompt, user_prompt,
              max_tokens, thinking=True):
    model.set_adapter(adapter_name)
    engine.SYSTEM_PROMPT = system_prompt
    engine.USER_PROMPT = user_prompt
    engine.ENABLE_THINKING = thinking
    images, prompts = [], []
    try:
        for row in rows:
            image = image_for(row)
            images.append(image)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": user_prompt}]},
            ]
            prompts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
            ))
        inputs = processor(text=prompts, images=images, padding=True, truncation=False,
                           return_tensors="pt").to("cuda:0")
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=processor.tokenizer.pad_token_id)
        new_ids = output[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(new_ids, skip_special_tokens=False)
        elapsed = (time.monotonic() - started) / max(1, len(rows))
        prefix = "<think>\n" if thinking else ""
        return [(parse_answer(prefix + text), prefix + text,
                 int(ids.shape[0]), elapsed) for ids, text in zip(new_ids, decoded)]
    finally:
        for image in images:
            image.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--router-adapter", required=True)
    p.add_argument("--router-checkpoint", default="")
    p.add_argument("--air-adapter", required=True); p.add_argument("--air-checkpoint", default="")
    p.add_argument("--ground-adapter", required=True); p.add_argument("--ground-checkpoint", default="")
    p.add_argument("--sea-adapter", required=True); p.add_argument("--sea-checkpoint", default="")
    p.add_argument("--router-prompt-parquet", type=Path)
    p.add_argument("--air-prompt-parquet", type=Path)
    p.add_argument("--ground-prompt-parquet", type=Path)
    p.add_argument("--sea-prompt-parquet", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--router-mode", choices=["thinking", "direct"], default="thinking")
    p.add_argument("--air-mode", choices=["thinking", "direct"], default="thinking")
    p.add_argument("--ground-mode", choices=["thinking", "direct"], default="thinking")
    p.add_argument("--sea-mode", choices=["thinking", "direct"], default="thinking")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit < 0:
        p.error("batch-size/max-new-tokens must be positive and limit must be >= 0")

    rows = load_items(args.data)
    if args.limit:
        rows = rows[:args.limit]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.is_file() and not args.no_resume:
        with output.open(newline="", encoding="utf-8") as f:
            done = {r["file_name"] for r in csv.DictReader(f) if r.get("file_name")}
    rows = [r for r in rows if r["file_name"] not in done]
    if not rows:
        print("Nothing to process")
        return

    router_prompt = load_prompt(args.router_prompt_parquet) or (ROUTER_SYSTEM, ROUTER_USER)
    specialist_prompts = {
        "Air": load_prompt(args.air_prompt_parquet) or (SPECIALIST_SYSTEM["Air"], SPECIALIST_USER["Air"]),
        "Ground": load_prompt(args.ground_prompt_parquet) or (SPECIALIST_SYSTEM["Ground"], SPECIALIST_USER["Ground"]),
        "Sea": load_prompt(args.sea_prompt_parquet) or (SPECIALIST_SYSTEM["Sea"], SPECIALIST_USER["Sea"]),
    }
    router_path = resolve_adapter(args.router_adapter, args.router_checkpoint)
    paths = {"air": resolve_adapter(args.air_adapter, args.air_checkpoint),
             "ground": resolve_adapter(args.ground_adapter, args.ground_checkpoint),
             "sea": resolve_adapter(args.sea_adapter, args.sea_checkpoint)}
    processor, model = load_model(router_path)
    add_adapters(model, paths)

    print(f"Routing {len(rows)} images with the 81-class adapter", flush=True)
    routed = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        results = run_batch(batch, processor, model, "router", router_prompt[0], router_prompt[1],
                            args.max_new_tokens, args.router_mode == "thinking")
        for row, (prediction, raw, tokens, seconds) in zip(batch, results):
            domain = next((d for d, prefix in DOMAIN_PREFIX.items() if prediction.startswith(prefix)), "Unknown")
            routed.append({"row": row, "router_prediction": prediction, "router_raw": raw,
                           "predicted_domain": domain, "router_tokens": tokens,
                           "router_seconds": round(seconds, 4)})
            print(f"[router {offset + len(routed) - offset}/{len(rows)}] {row['file_name']} -> {prediction} [{domain}]", flush=True)

    specialist_results = {}
    for domain in DOMAIN_CLASSES:
        group = [item for item in routed if item["predicted_domain"] == domain]
        if not group:
            continue
        print(f"Running {domain} specialist on {len(group)} images", flush=True)
        prompt = specialist_prompts[domain]
        adapter_name = DOMAIN_ADAPTER[domain]
        mode = getattr(args, f"{domain.lower()}_mode")
        specialist_results[domain] = []
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset:offset + args.batch_size]
            specialist_results[domain].extend(zip(batch, run_batch(
                [item["row"] for item in batch], processor, model, adapter_name,
                prompt[0], prompt[1], args.max_new_tokens, mode == "thinking")))

    fields = ["file_name", "ground_truth", "router_prediction", "predicted_domain", "selected_adapter",
              "specialist_prediction", "router_domain_correct", "final_correct", "router_seconds",
              "specialist_seconds", "router_raw", "specialist_raw"]
    mode = "w" if args.no_resume or not output.exists() else "a"
    with output.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w": writer.writeheader()
        for domain in DOMAIN_CLASSES:
            for item, result in specialist_results.get(domain, []):
                row = item["row"]; prediction, raw, _, seconds = result
                gt = row.get("ground_truth", "")
                domain_gt = next((d for d, prefix in DOMAIN_PREFIX.items() if gt.startswith(prefix)), "")
                writer.writerow({
                    "file_name": row["file_name"], "ground_truth": gt,
                    "router_prediction": item["router_prediction"], "predicted_domain": domain,
                    "selected_adapter": DOMAIN_ADAPTER[domain], "specialist_prediction": prediction,
                    "router_domain_correct": bool(domain_gt and domain_gt == domain),
                    "final_correct": bool(prediction.casefold() == gt.casefold()) if gt else "",
                    "router_seconds": item["router_seconds"], "specialist_seconds": round(seconds, 4),
                    "router_raw": item["router_raw"], "specialist_raw": raw,
                })
                f.flush()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
