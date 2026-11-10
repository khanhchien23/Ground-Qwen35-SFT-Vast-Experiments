#!/usr/bin/env python
"""Ground-only inference using the shared batched Qwen inference engine.

Prompts are read from the SFT train Parquet, not reconstructed from labels.
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

import inference_selected_no_errors as engine


def main():
    # Consume our additional argument, forwarding all shared CLI options.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt-parquet", type=Path)
    parser.add_argument("--mode", choices=["direct", "answer_only"], default="answer_only")
    args, remaining = parser.parse_known_args()
    if args.mode == "answer_only":
        if args.prompt_parquet is None:
            parser.error("--prompt-parquet is required for answer_only")
        prompts = pq.read_table(args.prompt_parquet, columns=["system_prompt", "user_prompt"])
        if not len(prompts):
            raise ValueError("Prompt Parquet is empty")
        engine.SYSTEM_PROMPT = prompts["system_prompt"][0].as_py()
        engine.USER_PROMPT = prompts["user_prompt"][0].as_py()

    original_loader = engine.items_from_data

    def ground_items(data):
        # HF snapshot symlinks resolve to extensionless blob files. This
        # entrypoint supports Parquet files or image directories, so inspect
        # the filesystem type rather than relying on the filename suffix.
        if data.is_file():
            table = pq.read_table(data, columns=["file_name", "gt_class", "image_bytes"])
            rows = [{"file_name": row["file_name"], "ground_truth": row["gt_class"],
                     "image_bytes": row["image_bytes"]} for row in table.to_pylist()]
        else:
            rows = original_loader(data)
        selected = [row for row in rows if row["ground_truth"].startswith("Grd.")]
        if not selected:
            raise ValueError("No Ground-only images found")
        return selected

    engine.items_from_data = ground_items
    sys.argv = [sys.argv[0]] + remaining + ["--mode", args.mode]
    engine.main()


if __name__ == "__main__":
    main()
