#!/usr/bin/env python3
"""
generate_llamaguard_dataset_unsloth.py
python generate_data.py `
  --meta-prompts eu_ai_act_dataset_v3.jsonl `
  --out-dir ./dataset `
  --model huihui-ai/Huihui-Qwen3-8B-abliterated-v2 `
  --batch-size 4 `
  --samples-per-meta 2 `
  --temperature-jitter 0.1 `
  --edge-policy keep `
  --verbose
Dataset generation using Unsloth's FastLanguageModel (local, no API).
Features:
- Batched inference via Unsloth (fast, low memory)
- Multiple samples per meta‑prompt (num_return_sequences)
- Temperature jitter, top_p, etc.
- Retry on parse failure (regenerate)
- Checkpointing (resume)
- Rich metadata
- Axolotl / LlamaFactory / Unsloth compatible output
"""
print("===== START OF SCRIPT =====")

import argparse
import json
import hashlib
import random
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter

import numpy as np
from tqdm import tqdm
import jsonschema
import torch
from unsloth import FastLanguageModel

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_MODEL = "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"  # or any HF model
TAXONOMY = {i: {"code": f"O{i}"} for i in range(1, 8)}
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in User messages.

User: {user}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
        "additionalProperties": False,
    }
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def hash_text(text: str) -> str:
    # Kept for checkpointing, but no longer used for dedup
    return hashlib.sha256(text.encode()).hexdigest()

def build_completion(prompt: str, group: int, level: str, edge_policy: str) -> Optional[Dict]:
    if level == "VIOLATION":
        comp = f"unsafe\n{TAXONOMY[group]['code']}<|eot_id|>"
        final_level = "VIOLATION"
    elif level == "COMPLIANT":
        comp = "safe<|eot_id|>"
        final_level = "COMPLIANT"
    else:  # EDGE_CASE
        if edge_policy == "drop":
            return None
        if edge_policy == "safe":
            comp = "safe<|eot_id|>"
            final_level = "COMPLIANT"
        elif edge_policy == "keep":
            # Preserve ambiguous examples as a distinct third training label.
            comp = "edge_case<|eot_id|>"
            final_level = "EDGE_CASE"
        else:  # unsafe
            comp = f"unsafe\n{TAXONOMY[group]['code']}<|eot_id|>"
            final_level = "VIOLATION"
    return {
        "prompt": PROMPT_TEMPLATE.format(user=prompt),
        "completion": comp,
        "original_group": group,
        "original_level": level,
        "final_level": final_level,
    }

def setup_logging(verbosity: int = 1):
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.INFO)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

# -----------------------------------------------------------------------------
# Main Generator (Unsloth)
# -----------------------------------------------------------------------------
class LlamaGuardDatasetGeneratorUnsloth:
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        temperature: float = 0.8,
        temperature_jitter: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 700,
        edge_policy: str = "unsafe",
        checkpoint_interval: int = 100,
        max_retries: int = 3,
        samples_per_meta: int = 1,
        batch_size: int = 4,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self.temperature_jitter = temperature_jitter
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.edge_policy = edge_policy
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.samples_per_meta = samples_per_meta
        self.batch_size = batch_size
        self.verbose = verbose

        # Load Unsloth model
        logging.info(f"Loading Unsloth model: {model_name} ...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,          # or False for full precision
            dtype=None,
        )
        FastLanguageModel.for_inference(self.model)
        logging.info("Unsloth model ready.")

        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.processed_meta_hashes: Set[str] = set()
        self.generated_examples: List[Dict] = []
        self.stats = {
            "total_meta_prompts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "level_counts": Counter(),
            "group_counts": Counter(),
        }
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                cp = json.load(f)
            self.processed_meta_hashes = set(cp.get("processed_meta_hashes", []))
            self.generated_examples = cp.get("generated_examples", [])
            logging.info(f"Resumed: {len(self.generated_examples)} examples, "
                         f"{len(self.processed_meta_hashes)} meta prompts processed.")
        else:
            logging.info("No checkpoint, starting fresh.")

    def _save_checkpoint(self):
        cp = {
            "processed_meta_hashes": list(self.processed_meta_hashes),
            "generated_examples": self.generated_examples,
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(cp, f, indent=2)

    def _generate_for_meta_prompt(self, meta_prompt: str, temperature: float) -> Optional[List[Dict]]:
        """Generate one or more completions using Unsloth, retry on parse failure."""
        messages = [{"role": "user", "content": meta_prompt}]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        for attempt in range(self.max_retries):
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_tokens,
                        temperature=temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        num_return_sequences=self.samples_per_meta,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                # Decode each generated sequence
                results = []
                for output in outputs:
                    prompt_len = inputs['input_ids'].shape[1]
                    new_tokens = output[prompt_len:]
                    raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    # Try to parse JSON
                    try:
                        data = json.loads(raw)
                        jsonschema.validate(data, OUTPUT_SCHEMA)
                        if isinstance(data, list):
                            results.extend(data)
                        else:
                            results.append(data)
                    except (json.JSONDecodeError, jsonschema.ValidationError):
                        # Try to salvage single object
                        try:
                            obj = json.loads(raw)
                            if isinstance(obj, dict) and "prompt" in obj:
                                results.append(obj)
                        except:
                            pass
                if results:
                    return results
            except Exception as e:
                logging.warning(f"Generation attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))
        return None

    def generate_from_meta_prompts(self, meta_prompts: List[Dict]):
        # Filter already processed
        new_meta = []
        for mp in meta_prompts:
            key = hash_text(mp["meta_prompt"])
            if key not in self.processed_meta_hashes:
                new_meta.append(mp)
        logging.info(f"Total: {len(meta_prompts)}, new: {len(new_meta)}")
        if not new_meta:
            return

        # Process in batches
        total = len(new_meta)
        pbar = tqdm(total=total, desc="Generating")
        for i in range(0, total, self.batch_size):
            batch = new_meta[i:i+self.batch_size]
            for mp in batch:
                meta_prompt_id = hash_text(mp["meta_prompt"])
                self.stats["total_meta_prompts"] += 1

                # Apply jitter
                if self.temperature_jitter > 0:
                    temp = self.temperature + random.uniform(-self.temperature_jitter, self.temperature_jitter)
                    temp = max(0.01, min(2.0, temp))
                else:
                    temp = self.temperature

                parsed = self._generate_for_meta_prompt(mp["meta_prompt"], temp)
                if parsed is None:
                    self.stats["failed_generations"] += 1
                else:
                    self.stats["successful_generations"] += 1
                    for item in parsed:
                        p = item["prompt"].strip()
                        if not p:
                            continue
                        # No deduplication – keep all prompts
                        example = build_completion(p, mp["group"], mp["level"], self.edge_policy)
                        if example is None:
                            continue
                        example["meta_prompt_id"] = meta_prompt_id
                        for key, value in mp.items():
                            if key not in ["meta_prompt", "group", "level"]:
                                example[key] = value
                        self.generated_examples.append(example)
                        final_level = example["final_level"]
                        self.stats["level_counts"][final_level] += 1
                        self.stats["group_counts"][mp["group"]] += 1

                self.processed_meta_hashes.add(meta_prompt_id)
                pbar.update(1)

                # Checkpoint
                if len(self.processed_meta_hashes) % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                    pbar.set_postfix({"examples": len(self.generated_examples)})

        pbar.close()
        self._save_checkpoint()
        logging.info("Generation complete.")

    def compute_statistics(self) -> Dict:
        stats = self.stats.copy()
        stats["total_examples"] = len(self.generated_examples)
        stats["level_distribution"] = dict(stats["level_counts"])
        stats["group_distribution"] = dict(stats["group_counts"])
        total = stats["total_examples"]
        stats["unsafe_ratio"] = stats["level_counts"].get("VIOLATION", 0) / total if total else 0.0
        stats["safe_ratio"] = stats["level_counts"].get("COMPLIANT", 0) / total if total else 0.0
        return stats

    def write_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        examples = self.generated_examples
        random.shuffle(examples)
        n = len(examples)
        n_val = int(val_ratio * n)
        n_test = int(test_ratio * n)
        n_train = n - n_val - n_test
        splits = {
            "train": examples[:n_train],
            "val": examples[n_train:n_train+n_val],
            "test": examples[n_train+n_val:]
        }
        for name, data in splits.items():
            out_path = self.output_dir / f"{name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logging.info(f"Wrote {len(data)} to {out_path}")

    def generate_quality_report(self, stats: Dict):
        report_lines = [
            "=" * 60,
            "Quality Report",
            f"Total meta prompts: {stats['total_meta_prompts']}",
            f"Successful: {stats['successful_generations']}",
            f"Failed: {stats['failed_generations']}",
            f"Examples: {stats['total_examples']}",
            f"Unsafe: {stats['unsafe_ratio']:.2%}, Safe: {stats['safe_ratio']:.2%}",
            "\nLevel distribution:"
        ]
        for level, count in stats["level_distribution"].items():
            report_lines.append(f"  {level}: {count}")
        report_lines.append("\nGroup distribution:")
        for group, count in stats["group_distribution"].items():
            report_lines.append(f"  O{group}: {count}")
        report_text = "\n".join(report_lines)
        print(report_text)
        (self.output_dir / "quality_report.txt").write_text(report_text)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    print(">>> ENTERED MAIN <<<")
    parser = argparse.ArgumentParser(description="Generate dataset with Unsloth.")
    parser.add_argument("--meta-prompts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--temperature-jitter", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--edge-policy", choices=["unsafe", "safe", "keep", "drop"], default="unsafe")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-meta", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    setup_logging(verbosity=2 if args.verbose else 1)

    meta_prompts = []
    with open(args.meta_prompts, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                meta_prompts.append(json.loads(line))
    logging.info(f"Loaded {len(meta_prompts)} meta prompts.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = LlamaGuardDatasetGeneratorUnsloth(
        model_name=args.model,
        output_dir=out_dir,
        temperature=args.temperature,
        temperature_jitter=args.temperature_jitter,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        edge_policy=args.edge_policy,
        checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries,
        samples_per_meta=args.samples_per_meta,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    if args.no_resume:
        if generator.checkpoint_file.exists():
            generator.checkpoint_file.unlink()
        generator.processed_meta_hashes.clear()
        generator.generated_examples.clear()

    generator.generate_from_meta_prompts(meta_prompts)
    stats = generator.compute_statistics()
    generator.generate_quality_report(stats)
    generator.write_splits()
    logging.info("Done.")

if __name__ == "__main__":
    main()