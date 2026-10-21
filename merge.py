from pathlib import Path
import json

# Run this script from the "eu guard" folder.
ROOT = Path(__file__).resolve().parent

SOURCE_FOLDERS = [
    ROOT / "gemini_dataset_10k",
    ROOT / "gemini_dataset_20k",
    ROOT / "qwen_dataset_10k",
]

OUTPUT_DIR = ROOT / "merged_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_MERGE = [
    "qwen_train.jsonl",
    "qwen_val.jsonl",
    "qwen_test.jsonl",
    "dpo_train.jsonl",
    "dpo_val.jsonl",
    "dpo_test.jsonl",
]

def validate_jsonl_line(line: str, file_path: Path, line_number: int) -> None:
    try:
        json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {file_path}, line {line_number}: {exc}"
        ) from exc

for filename in FILES_TO_MERGE:
    output_path = OUTPUT_DIR / filename.replace(".jsonl", "_combined.jsonl")
    total_lines = 0
    sources_found = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for folder in SOURCE_FOLDERS:
            source_path = folder / filename
            if not source_path.exists():
                print(f"[SKIP] Missing: {source_path}")
                continue

            sources_found += 1
            source_count = 0
            with source_path.open("r", encoding="utf-8-sig") as src:
                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    validate_jsonl_line(line, source_path, line_number)
                    out.write(line.rstrip("\r\n") + "\n")
                    source_count += 1
                    total_lines += 1

            print(f"[OK] {source_path.name} from {folder.name}: {source_count:,} records")

    if sources_found:
        print(f"[DONE] {output_path} — {total_lines:,} total records")
    else:
        output_path.unlink(missing_ok=True)
        print(f"[WARN] No source files found for {filename}; no output created.")

print(f"\nFinished. Combined files are in: {OUTPUT_DIR}")
print("Note: records are concatenated as-is; duplicates are not removed.")
