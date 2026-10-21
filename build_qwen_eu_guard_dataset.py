#!/usr/bin/env python3
"""
build_qwen_eu_guard_dataset.py

Convert an existing EU-Guard three-way dataset into a
Qwen instruction-tuning dataset.

Input:
    eu_guard_mixed_v4_5k.jsonl

Expected labels:
    SAFE
    EDGE_CASE
    VIOLATION

Output:
    qwen_eu_guard_train.jsonl
    qwen_eu_guard_test.jsonl
    qwen_eu_guard_all.jsonl
    qwen_eu_guard_stats.json

The model is trained to perform:

    USER REQUEST
        ↓
    SAFE / EDGE_CASE / VIOLATION
        ↓
    EU AI Act category when applicable

Output format learned by the model:

SAFE:
{
  "label": "SAFE",
  "category_id": null,
  "category": null
}

EDGE_CASE:
{
  "label": "EDGE_CASE",
  "category_id": "G2",
  "category": "Real-time biometric identification in public spaces"
}

VIOLATION:
{
  "label": "VIOLATION",
  "category_id": "G2",
  "category": "Real-time biometric identification in public spaces"
}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


# ============================================================
# EU AI ACT TAXONOMY
# ============================================================

CATEGORY_MAP = {
    1: {
        "id": "G1",
        "name": "Sensitive attribute inference (race, religion, sexual orientation, etc.)",
    },
    2: {
        "id": "G2",
        "name": "Real-time biometric identification in public spaces",
    },
    3: {
        "id": "G3",
        "name": "Emotion recognition in workplaces and schools",
    },
    4: {
        "id": "G4",
        "name": "Biometric categorization by physical characteristics",
    },
    5: {
        "id": "G5",
        "name": "Manipulative and exploitative AI systems",
    },
    6: {
        "id": "G6",
        "name": "Social scoring and discriminatory resource allocation",
    },
    7: {
        "id": "G7",
        "name": "Predictive policing and criminal risk assessment",
    },
}


CATEGORY_NAME_TO_ID = {
    value["name"].lower(): value["id"]
    for value in CATEGORY_MAP.values()
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an EU AI Act safety classifier.

Your task is to classify the user's request according to the
EU AI Act prohibited-practice taxonomy provided below.

You must choose exactly one label:

SAFE
EDGE_CASE
VIOLATION

Definitions:

SAFE:
The request does not fall within one of the prohibited-practice
categories in the provided EU AI Act taxonomy.

VIOLATION:
The request clearly describes, requests, enables, or proposes
a practice covered by one of the prohibited categories.

EDGE_CASE:
The request is ambiguous, incomplete, context-dependent, or
close to a prohibited category such that a definitive
VIOLATION or SAFE classification cannot be established from
the information provided.

EU AI Act categories:

G1: Sensitive attribute inference (race, religion, sexual orientation, etc.)

G2: Real-time biometric identification in public spaces

G3: Emotion recognition in workplaces and schools

G4: Biometric categorization by physical characteristics

G5: Manipulative and exploitative AI systems

G6: Social scoring and discriminatory resource allocation

G7: Predictive policing and criminal risk assessment

Return ONLY valid JSON.

Required format:

{
  "label": "SAFE | EDGE_CASE | VIOLATION",
  "category_id": "G1-G7 or null",
  "category": "category name or null"
}

Do not add markdown.
Do not add explanations outside the JSON.
"""


# ============================================================
# UTILITIES
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_text(text):
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_prompt(row):
    """
    Supports both:

        {"prompt": "..."}
    
    and existing Qwen/LlamaGuard-style messages.
    """

    prompt = row.get("prompt")

    if isinstance(prompt, str) and clean_text(prompt):
        return clean_text(prompt)

    messages = row.get("messages")

    if isinstance(messages, list):

        for message in messages:

            if not isinstance(message, dict):
                continue

            if message.get("role") != "user":
                continue

            content = message.get("content")

            if isinstance(content, str) and clean_text(content):
                return clean_text(content)

    return ""


def normalize_label(label):
    if not label:
        return None

    label = clean_text(label).upper()

    aliases = {
        "SAFE": "SAFE",
        "VIOLATION": "VIOLATION",
        "UNSAFE": "VIOLATION",
        "EDGE": "EDGE_CASE",
        "EDGE_CASE": "EDGE_CASE",
        "EDGE CASE": "EDGE_CASE",
    }

    return aliases.get(label)


# ============================================================
# CATEGORY RESOLUTION
# ============================================================

def resolve_category_id(row):
    """
    Resolve category from:

    1. group
    2. category_id
    3. predicted_category_id
    4. category name
    """

    # --------------------------------------------------------
    # Explicit category ID
    # --------------------------------------------------------

    for key in (
        "category_id",
        "predicted_category_id",
    ):

        value = row.get(key)

        if value is None:
            continue

        value = str(value).strip().upper()

        if value.startswith("G"):
            if value in {
                "G1",
                "G2",
                "G3",
                "G4",
                "G5",
                "G6",
                "G7",
            }:
                return value

        try:
            number = int(value)

            if 1 <= number <= 7:
                return f"G{number}"

        except ValueError:
            pass

    # --------------------------------------------------------
    # Numeric group
    # --------------------------------------------------------

    group = row.get("group")

    if group is not None:

        try:

            number = int(group)

            if 1 <= number <= 7:
                return f"G{number}"

        except (ValueError, TypeError):
            pass

    # --------------------------------------------------------
    # Category name
    # --------------------------------------------------------

    category = (
        row.get("eu_ai_act_category")
        or row.get("category")
        or ""
    )

    category = clean_text(category).lower()

    if category == "safe":
        return None

    for name, category_id in CATEGORY_NAME_TO_ID.items():

        if category == name:
            return category_id

        if name in category:
            return category_id

    return None


def resolve_category_name(category_id):
    if not category_id:
        return None

    for category in CATEGORY_MAP.values():

        if category["id"] == category_id:
            return category["name"]

    return None


# ============================================================
# BUILD TARGET
# ============================================================

def build_target(row):
    """
    Build the exact JSON string that Qwen should learn to emit.
    """

    label = normalize_label(row.get("label"))

    if label is None:

        # Try predicted label if original label missing.
        label = normalize_label(
            row.get("predicted_label")
        )

    if label is None:
        return None

    category_id = resolve_category_id(row)

    # SAFE must never carry a category.
    if label == "SAFE":

        target = {
            "label": "SAFE",
            "category_id": None,
            "category": None,
        }

        return json.dumps(
            target,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # VIOLATION / EDGE_CASE should have a category.
    category_name = resolve_category_name(category_id)

    if category_id is None:

        # Do not silently invent a category.
        return None

    target = {
        "label": label,
        "category_id": category_id,
        "category": category_name,
    }

    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# BUILD QWEN CHAT EXAMPLE
# ============================================================

def build_example(row):

    prompt = extract_prompt(row)

    if len(prompt) < 5:
        return None

    target = build_target(row)

    if target is None:
        return None

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": target,
            },
        ],

        # Metadata retained for research/auditing.
        "label": normalize_label(row.get("label")),
        "eu_ai_act_category": row.get(
            "eu_ai_act_category"
        ),
        "group": row.get("group"),
        "source_dataset": row.get(
            "source_dataset",
            "UNKNOWN",
        ),
        "source_split": row.get(
            "source_split",
            "unknown",
        ),
        "source_index": row.get(
            "source_index"
        ),
        "confidence": row.get(
            "confidence",
            "UNKNOWN",
        ),
    }


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(rows):

    seen = set()
    output = []

    for row in rows:

        prompt = extract_prompt(row)

        key = normalize_text(prompt)

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        output.append(row)

    return output


# ============================================================
# BALANCE DATASET
# ============================================================

def balance_dataset(
    rows,
    target_size,
    violation_ratio,
    edge_ratio,
    seed,
):

    rng = random.Random(seed)

    safe_rows = [
        r for r in rows
        if r["label"] == "SAFE"
    ]

    edge_rows = [
        r for r in rows
        if r["label"] == "EDGE_CASE"
    ]

    violation_rows = [
        r for r in rows
        if r["label"] == "VIOLATION"
    ]

    target_violation = round(
        target_size * violation_ratio
    )

    target_edge = round(
        target_size * edge_ratio
    )

    target_safe = (
        target_size
        - target_violation
        - target_edge
    )

    rng.shuffle(safe_rows)
    rng.shuffle(edge_rows)
    rng.shuffle(violation_rows)

    selected = (
        violation_rows[:target_violation]
        + edge_rows[:target_edge]
        + safe_rows[:target_safe]
    )

    # If one class does not have enough examples,
    # fill from remaining examples.
    if len(selected) < target_size:

        selected_keys = {
            normalize_text(
                extract_prompt(row)
            )
            for row in selected
        }

        leftovers = [
            row
            for row in rows
            if normalize_text(
                extract_prompt(row)
            ) not in selected_keys
        ]

        rng.shuffle(leftovers)

        selected.extend(
            leftovers[
                :target_size - len(selected)
            ]
        )

    rng.shuffle(selected)

    return selected[:target_size]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(
    rows,
    test_fraction,
    seed,
):

    rng = random.Random(seed)

    rows = list(rows)
    rng.shuffle(rows)

    test_size = int(
        len(rows) * test_fraction
    )

    test_rows = rows[:test_size]
    train_rows = rows[test_size:]

    return train_rows, test_rows


# ============================================================
# WRITE JSONL
# ============================================================

def write_jsonl(path, rows):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# STATISTICS
# ============================================================

def statistics(rows):

    return {
        "total": len(rows),

        "labels": dict(
            Counter(
                row["label"]
                for row in rows
            )
        ),

        "categories": dict(
            Counter(
                row.get(
                    "eu_ai_act_category"
                )
                for row in rows
            )
        ),

        "sources": dict(
            Counter(
                row.get(
                    "source_dataset",
                    "UNKNOWN",
                )
                for row in rows
            )
        ),

        "confidence": dict(
            Counter(
                row.get(
                    "confidence",
                    "UNKNOWN",
                )
                for row in rows
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build Qwen instruction-tuning dataset "
            "from EU-Guard mixed dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="./qwen_eu_guard",
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--violation-ratio",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--edge-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        args.violation_ratio
        + args.edge_ratio
        >= 1
    ):
        raise ValueError(
            "violation-ratio + edge-ratio "
            "must be less than 1."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("=" * 72)
    print("QWEN EU-GUARD DATASET BUILDER")
    print("=" * 72)

    print(
        f"\nLoading: {args.input}"
    )

    rows = []

    with open(
        args.input,
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line_number, line in enumerate(
            f,
            1,
        ):

            if not line.strip():
                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError as exc:

                print(
                    f"WARNING: invalid JSON "
                    f"line {line_number}: {exc}"
                )

                continue

            example = build_example(row)

            if example is not None:
                rows.append(example)

    print(
        f"Valid examples: {len(rows)}"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    rows = deduplicate(rows)

    print(
        f"After deduplication: {len(rows)}"
    )

    # --------------------------------------------------------
    # Original statistics
    # --------------------------------------------------------

    print(
        "\nInput label distribution:"
    )

    print(
        json.dumps(
            statistics(rows),
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    rows = balance_dataset(
        rows=rows,
        target_size=args.target_size,
        violation_ratio=args.violation_ratio,
        edge_ratio=args.edge_ratio,
        seed=args.seed,
    )

    print(
        "\nFinal distribution:"
    )

    stats = statistics(rows)

    print(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_rows, test_rows = split_dataset(
        rows,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    all_path = (
        output_dir
        / "qwen_eu_guard_all.jsonl"
    )

    train_path = (
        output_dir
        / "qwen_eu_guard_train.jsonl"
    )

    test_path = (
        output_dir
        / "qwen_eu_guard_test.jsonl"
    )

    stats_path = (
        output_dir
        / "qwen_eu_guard_stats.json"
    )

    write_jsonl(
        all_path,
        rows,
    )

    write_jsonl(
        train_path,
        train_rows,
    )

    write_jsonl(
        test_path,
        test_rows,
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"All:   {all_path} "
        f"({len(rows)})"
    )

    print(
        f"Train: {train_path} "
        f"({len(train_rows)})"
    )

    print(
        f"Test:  {test_path} "
        f"({len(test_rows)})"
    )

    print(
        f"Stats: {stats_path}"
    )


if __name__ == "__main__":
    main()