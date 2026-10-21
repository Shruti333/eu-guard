#!/usr/bin/env python3
"""
EU-GUARD MIXED DATASET BUILDER V3 (THREE-WAY)

Builds a balanced EU-Guard fine-tuning dataset from:
  1. EU synthetic dataset (eu_ai_act_dataset_v3.jsonl)
  2. Anthropic/hh-rlhf
  3. PKU-Alignment/BeaverTails

Three-way version with similarity + category-margin boundary handling: SAFE / VIOLATION / EDGE_CASE.

Main improvements over V1:
  - Uses existing EU synthetic prompts as category anchors.
  - Uses top-k semantic similarity per category instead of one global maximum.
  - Has HIGH / MEDIUM / SAFE confidence zones.
  - Enforces per-category violation quotas.
  - Explicitly balances data sources.
  - Uses medium-confidence real examples before falling back to synthetic examples.
  - Keeps source provenance and similarity metadata.
  - Performs prompt-level deduplication.
  - Splits train/test after dataset construction.

Important:
This is a semantic data-curation pipeline, not a legal annotation oracle.
The generated mapping should be audited before publication/training.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


SYSTEM_PROMPT = (
    "You are an AI safety classifier. Classify the user request according to "
    "the EU AI Act taxonomy provided during fine-tuning."
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def clean(value):
    return re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()


def normalize_prompt(text):
    text = clean(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def row_prompt(row):
    """
    Flexible extraction because the synthetic JSONL may contain prompt,
    instruction, question, text, or chat/messages-style rows.
    """
    for key in ("prompt", "instruction", "question", "input", "text"):
        value = row.get(key)
        if isinstance(value, str) and clean(value):
            return clean(value)

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and clean(content):
                    return clean(content)

    return ""


def extract_rule_from_meta(text, fallback):
    """
    Extract useful rule text from a meta-prompt while avoiding embedding the
    entire generation template as a category anchor.
    """
    text = str(text or "")

    patterns = [
        r"THE RULE\s*:?\s*(.*?)(?=\n\s*(?:Your task|Constraints|Output format|Each prompt|Valid sources)|\Z)",
        r"Rule\s*:?\s*(.*?)(?=\n\s*(?:Your task|Constraints|Output format|Each prompt)|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.I | re.S)
        if match:
            result = clean(match.group(1))
            if result:
                return result[:2000]

    return clean(text)[:1200] or fallback


# ---------------------------------------------------------------------
# EU taxonomy + synthetic anchor loading
# ---------------------------------------------------------------------

def load_eu_reference(path, anchors_per_category, seed):
    """
    Reads the user's eu_ai_act_dataset_v3.jsonl.

    Expected taxonomy fields:
      level
      group
      group_label
      meta_prompt

    The file may additionally contain actual generated prompts. Those are used
    as semantic anchors if present.
    """

    rng = random.Random(seed)

    category_names = {}
    category_groups = {}
    rule_anchors = defaultdict(list)
    prompt_anchors = defaultdict(list)
    synthetic_rows = []

    print(f"\nLoading EU reference: {path}")

    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping invalid JSON at line {line_number}: {exc}")
                continue

            level = clean(row.get("level")).upper()

            # Taxonomy rows.
            if level == "VIOLATION":
                group = row.get("group")
                group_label = clean(row.get("group_label"))

                if group is not None and group_label:
                    category_id = f"G{group}"

                    category_names[category_id] = group_label
                    category_groups[category_id] = group

                    rule = extract_rule_from_meta(
                        row.get("meta_prompt"),
                        group_label,
                    )

                    if rule:
                        rule_anchors[category_id].append(
                            f"EU category: {group_label}. Rule: {rule}"
                        )

                    prompt = row_prompt(row)
                    if prompt:
                        prompt_anchors[category_id].append(
                            f"Request: {prompt}"
                        )
                        synthetic_rows.append({
                            "prompt": prompt,
                            "source_dataset": "EU_SYNTHETIC",
                            "source_split": "reference",
                            "source_index": line_number,
                            "category_id": category_id,
                            "label": "VIOLATION",
                            "original_row": row,
                        })

            # Also allow rows that directly contain group/category information
            # even if level is missing or differently named.
            group = row.get("group")
            group_label = clean(row.get("group_label"))

            if group is not None and group_label and level != "VIOLATION":
                category_id = f"G{group}"
                category_names.setdefault(category_id, group_label)
                category_groups.setdefault(category_id, group)

                prompt = row_prompt(row)
                if prompt:
                    prompt_anchors[category_id].append(f"Request: {prompt}")

                    if level in ("VIOLATION", "UNSAFE", "HARMFUL"):
                        synthetic_rows.append({
                            "prompt": prompt,
                            "source_dataset": "EU_SYNTHETIC",
                            "source_split": "reference",
                            "source_index": line_number,
                            "category_id": category_id,
                            "label": "VIOLATION",
                            "original_row": row,
                        })

    if not category_names:
        raise RuntimeError(
            "No EU violation categories discovered. "
            "Expected fields such as level='VIOLATION', group and group_label."
        )

    # Deduplicate anchors and cap prompt anchors per category.
    category_anchor_texts = {}

    for category_id in category_names:
        rules = list(dict.fromkeys(rule_anchors[category_id]))
        prompts = list(dict.fromkeys(prompt_anchors[category_id]))

        rng.shuffle(prompts)
        prompts = prompts[:anchors_per_category]

        anchors = rules + prompts

        if not anchors:
            anchors = [
                f"EU category: {category_names[category_id]}"
            ]

        category_anchor_texts[category_id] = anchors

    # Deduplicate synthetic rows.
    synthetic_rows = deduplicate_rows(synthetic_rows)

    print(f"Discovered {len(category_names)} EU violation categories.")
    for cid in sorted(category_names, key=lambda x: int(x[1:])):
        print(
            f"  {cid}: {category_names[cid]} | "
            f"anchors={len(category_anchor_texts[cid])} | "
            f"synthetic_rows={sum(1 for x in synthetic_rows if x['category_id'] == cid)}"
        )

    return (
        category_anchor_texts,
        category_names,
        category_groups,
        synthetic_rows,
    )


# ---------------------------------------------------------------------
# HH-RLHF
# ---------------------------------------------------------------------

def extract_hh_prompt(row):
    for field in ("chosen", "rejected"):
        text = str(row.get(field) or "")

        matches = re.findall(
            r"(?:^|\n)Human:\s*(.*?)(?=\nAssistant:|$)",
            text,
            re.I | re.S,
        )

        if matches:
            prompt = clean(matches[-1])
            if prompt:
                return prompt

        if "Assistant:" in text:
            candidate = text.split("Assistant:", 1)[0]
            candidate = re.sub(
                r"^\s*Human:\s*",
                "",
                candidate,
                flags=re.I,
            )
            candidate = clean(candidate)

            if candidate:
                return candidate

    for key in ("prompt", "question", "instruction", "text"):
        value = row.get(key)
        if isinstance(value, str) and clean(value):
            return clean(value)

    return ""


def load_hh(limit, seed):
    print("\nLoading Anthropic/hh-rlhf...")

    dataset = load_dataset("Anthropic/hh-rlhf")

    rng = random.Random(seed)
    output = []

    for split_name, split in dataset.items():
        indices = list(range(len(split)))
        rng.shuffle(indices)

        for index in indices:
            prompt = extract_hh_prompt(split[index])

            if len(prompt) >= 10:
                output.append({
                    "prompt": prompt,
                    "source_dataset": "Anthropic/hh-rlhf",
                    "source_split": split_name,
                    "source_index": int(index),
                })

            if len(output) >= limit:
                return output

    return output


# ---------------------------------------------------------------------
# BeaverTails
# ---------------------------------------------------------------------

def extract_beaver_prompt(row):
    for key in (
        "prompt",
        "question",
        "instruction",
        "input",
        "text",
    ):
        value = row.get(key)
        if isinstance(value, str) and clean(value):
            return clean(value)

    string_values = [
        clean(value)
        for value in row.values()
        if isinstance(value, str) and len(clean(value)) >= 10
    ]

    return max(string_values, key=len) if string_values else ""

def normalize_for_dedupe(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text
def load_beaver(limit, seed):
    candidates = (
        "PKU-Alignment/BeaverTails",
        "PKU-Alignment/BeaverTails-Evaluation",
    )

    last_error = None

    for dataset_name in candidates:
        try:
            print(f"\nLoading {dataset_name}...")
            dataset = load_dataset(dataset_name)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"Could not load BeaverTails. Last error: {last_error}"
        )

    rng = random.Random(seed + 17)
    output = []

    for split_name, split in dataset.items():
        indices = list(range(len(split)))
        rng.shuffle(indices)

        for index in indices:
            row = split[index]
            prompt = extract_beaver_prompt(row)

            if len(prompt) >= 10:
                output.append({
                    "prompt": prompt,
                    "source_dataset": "PKU-Alignment/BeaverTails",
                    "source_split": split_name,
                    "source_index": int(index),
                })

            if len(output) >= limit:
                return output

    return output


# ---------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------

def deduplicate_rows(rows):
    seen = set()
    output = []

    for row in rows:
        prompt = clean(row.get("prompt"))

        if len(prompt) < 10:
            continue

        key = normalize_prompt(prompt)

        if key in seen:
            continue

        seen.add(key)

        copied = dict(row)
        copied["prompt"] = prompt

        output.append(copied)

    return output


# ---------------------------------------------------------------------
# Category embedding and top-k classification
# ---------------------------------------------------------------------

def build_category_anchor_embeddings(
    model,
    category_anchor_texts,
):
    flat_texts = []
    flat_categories = []

    for category_id, anchors in category_anchor_texts.items():
        for anchor in anchors:
            flat_texts.append(anchor)
            flat_categories.append(category_id)

    print(
        f"\nEncoding {len(flat_texts)} EU category anchors..."
    )

    embeddings = model.encode(
        flat_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    category_to_indices = defaultdict(list)

    for index, category_id in enumerate(flat_categories):
        category_to_indices[category_id].append(index)

    return embeddings, category_to_indices


def classify_candidates(
    model,
    rows,
    anchor_embeddings,
    category_to_indices,
    high_threshold,
    medium_threshold,
    margin_threshold,
    edge_low_threshold,
    top_k,
):
    prompts = [row["prompt"] for row in rows]

    print(f"\nEncoding {len(prompts)} candidate prompts...")

    prompt_embeddings = model.encode(
        prompts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    similarities = prompt_embeddings @ anchor_embeddings.T

    output = []

    category_ids = list(category_to_indices.keys())

    for row_index, row in enumerate(rows):
        category_scores = {}

        for category_id in category_ids:
            indices = category_to_indices[category_id]
            values = similarities[row_index, indices]

            k = min(top_k, len(values))

            if k <= 0:
                score = -1.0
            else:
                top_values = np.partition(
                    values,
                    len(values) - k,
                )[-k:]

                score = float(np.mean(top_values))

            category_scores[category_id] = score

        ranked = sorted(
            category_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_category, best_score = ranked[0]

        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        margin = float(best_score - second_score)

        # Three-way boundary logic:
        # HIGH: very strong semantic match.
        # MEDIUM: moderate match; a clear category margin is treated as a
        # plausible VIOLATION, while a small margin is genuinely ambiguous.
        # EDGE_LOW..MEDIUM: near-boundary examples become EDGE_CASE.
        if best_score >= high_threshold:
            confidence = "HIGH"
            predicted_label = "VIOLATION"
        elif best_score >= medium_threshold:
            confidence = "MEDIUM"
            predicted_label = (
                "VIOLATION" if margin >= margin_threshold else "EDGE_CASE"
            )
        elif best_score >= edge_low_threshold:
            confidence = "BOUNDARY"
            predicted_label = "EDGE_CASE"
        else:
            confidence = "LOW"
            predicted_label = "SAFE"

        copied = dict(row)

        copied.update({
            "predicted_category_id": best_category,
            "best_score": float(best_score),
            "second_score": float(second_score),
            "score_margin": margin,
            "confidence": confidence,
            "predicted_label": predicted_label,
        })

        output.append(copied)

    return output


# ---------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------

def take_from_pool(pool, amount, rng, key=None):
    if amount <= 0 or not pool:
        return []

    copied = list(pool)

    if key is None:
        rng.shuffle(copied)
    else:
        copied.sort(key=key, reverse=True)

    return copied[:amount]


def allocate_integer(total, weights):
    """
    Allocate integer counts proportional to weights.
    """
    if total <= 0:
        return {key: 0 for key in weights}

    total_weight = sum(max(0, value) for value in weights.values())

    if total_weight <= 0:
        equal = total // len(weights)
        remainder = total % len(weights)

        result = {
            key: equal + (index < remainder)
            for index, key in enumerate(weights)
        }

        return result

    raw = {
        key: total * max(0, value) / total_weight
        for key, value in weights.items()
    }

    result = {
        key: int(value)
        for key, value in raw.items()
    }

    remainder = total - sum(result.values())

    fractions = sorted(
        (
            (raw[key] - result[key], key)
            for key in weights
        ),
        reverse=True,
    )

    for _, key in fractions[:remainder]:
        result[key] += 1

    return result


def select_violations(
    classified_rows,
    synthetic_rows,
    category_names,
    violation_target,
    high_ratio,
    medium_ratio,
    synthetic_ratio,
    rng,
):
    """
    Per-category quota.

    Order of preference:
      1. HIGH confidence real examples
      2. EU synthetic examples

    MEDIUM-confidence real examples are reserved as EDGE_CASE.

    This guarantees coverage even when HH-RLHF / BeaverTails contain
    few direct matches.
    """

    category_ids = list(category_names.keys())

    quota_weights = {
        category_id: 1.0
        for category_id in category_ids
    }

    quotas = allocate_integer(
        violation_target,
        quota_weights,
    )

    high_by_category = defaultdict(list)
    medium_by_category = defaultdict(list)
    synthetic_by_category = defaultdict(list)

    for row in classified_rows:
        cid = row["predicted_category_id"]

        if (
            row["predicted_label"] == "VIOLATION"
            and row.get("confidence") == "HIGH"
        ):
            high_by_category[cid].append(row)

        elif (
            row["predicted_label"] == "VIOLATION"
            and row.get("confidence") == "MEDIUM"
        ):
            medium_by_category[cid].append(row)

    for row in synthetic_rows:
        synthetic_by_category[row["category_id"]].append(row)

    for pool in high_by_category.values():
        pool.sort(key=lambda x: x["best_score"], reverse=True)

    for pool in medium_by_category.values():
        pool.sort(key=lambda x: x["best_score"], reverse=True)

    for pool in synthetic_by_category.values():
        rng.shuffle(pool)

    selected = []
    selection_report = {}

    for category_id in category_ids:
        quota = quotas[category_id]

        high_target = round(quota * high_ratio)
        medium_target = round(quota * medium_ratio)

        # Remaining quota is reserved for synthetic.
        synthetic_target = quota - high_target - medium_target

        chosen = []

        high_pool = high_by_category[category_id]
        chosen += high_pool[:high_target]

        remaining = quota - len(chosen)

        medium_pool = medium_by_category[category_id]

        # First take planned medium examples.
        planned_medium = min(
            len(medium_pool),
            min(medium_target, remaining),
        )

        chosen += medium_pool[:planned_medium]

        remaining = quota - len(chosen)

        # If high pool was smaller than expected, allow more medium.
        if remaining > 0:
            already_medium = {
                (
                    x.get("source_dataset"),
                    x.get("source_split"),
                    x.get("source_index"),
                )
                for x in chosen
            }

            extra_medium = [
                x
                for x in medium_pool
                if (
                    x.get("source_dataset"),
                    x.get("source_split"),
                    x.get("source_index"),
                ) not in already_medium
            ]

            chosen += extra_medium[:remaining]

        remaining = quota - len(chosen)

        # Fill remaining quota from true EU synthetic examples.
        if remaining > 0:
            synthetic_pool = synthetic_by_category[category_id]

            chosen += synthetic_pool[:remaining]

        selection_report[category_id] = {
            "quota": quota,
            "selected": len(chosen),
            "high_real": sum(
                1
                for x in chosen
                if x.get("confidence") == "HIGH"
            ),
            "medium_real": sum(
                1
                for x in chosen
                if x.get("confidence") == "MEDIUM"
            ),
            "synthetic": sum(
                1
                for x in chosen
                if x.get("source_dataset") == "EU_SYNTHETIC"
            ),
        }

        selected.extend(chosen[:quota])

    # If synthetic source did not contain enough rows, fill only from
    # remaining real candidates, preserving category labels.
    if len(selected) < violation_target:
        used = {
            (
                row.get("source_dataset"),
                row.get("source_split"),
                row.get("source_index"),
                row.get("prompt"),
            )
            for row in selected
        }

        fallback = [
            row
            for row in classified_rows
            if row["predicted_label"] == "VIOLATION"
            and (
                row.get("source_dataset"),
                row.get("source_split"),
                row.get("source_index"),
                row.get("prompt"),
            ) not in used
        ]

        fallback.sort(
            key=lambda x: (
                x["best_score"],
                x["score_margin"],
            ),
            reverse=True,
        )

        selected.extend(
            fallback[:violation_target - len(selected)]
        )

    rng.shuffle(selected)

    return selected[:violation_target], quotas, selection_report


def select_edge_cases(
    classified_rows,
    edge_target,
    rng,
):
    """Select genuine ambiguous real examples classified in the MEDIUM band.

    EDGE_CASE examples preserve uncertainty instead of being collapsed into
    SAFE or VIOLATION. They retain the closest EU category as metadata.
    """
    pool = [
        row for row in classified_rows
        if row["predicted_label"] == "EDGE_CASE"
    ]

    # Prefer examples closest to the violation boundary.
    pool.sort(
        key=lambda x: (x["best_score"], -x["score_margin"]),
        reverse=True,
    )

    selected = pool[:edge_target]
    rng.shuffle(selected)
    return selected


def select_safe(
    classified_rows,
    safe_target,
    hard_negative_fraction,
    rng,
):
    """
    SAFE selection contains:
      - ordinary low-score safe examples
      - near-boundary safe examples (hard negatives)

    Near-boundary examples are useful because they are semantically closer to
    EU prohibited categories but remain below the medium threshold.
    """

    safe_rows = [
        row
        for row in classified_rows
        if row["predicted_label"] == "SAFE"
    ]

    hard_count = round(safe_target * hard_negative_fraction)
    ordinary_count = safe_target - hard_count

    # Highest scoring SAFE examples are the hardest negatives.
    ranked = sorted(
        safe_rows,
        key=lambda x: x["best_score"],
        reverse=True,
    )

    hard = ranked[:hard_count]

    used = {
        (
            row.get("source_dataset"),
            row.get("source_split"),
            row.get("source_index"),
        )
        for row in hard
    }

    ordinary_pool = [
        row
        for row in safe_rows
        if (
            row.get("source_dataset"),
            row.get("source_split"),
            row.get("source_index"),
        ) not in used
    ]

    rng.shuffle(ordinary_pool)

    ordinary = ordinary_pool[:ordinary_count]

    selected = hard + ordinary

    # If insufficient, use any remaining safe rows.
    if len(selected) < safe_target:
        remaining = [
            row
            for row in safe_rows
            if row not in selected
        ]

        rng.shuffle(remaining)

        selected.extend(
            remaining[:safe_target - len(selected)]
        )

    rng.shuffle(selected)

    return selected[:safe_target]


# ---------------------------------------------------------------------
# Source balancing
# ---------------------------------------------------------------------

def rebalance_sources(rows, target_size, source_weights, rng):
    """
    Best-effort source balancing without inventing data.

    This is used only after semantic/category selection. If a requested source
    lacks enough examples, the available data is retained and remaining places
    are filled from the strongest unused examples.
    """

    by_source = defaultdict(list)

    for row in rows:
        by_source[row.get("source_dataset", "UNKNOWN")].append(row)

    desired = allocate_integer(target_size, source_weights)

    selected = []

    for source, target in desired.items():
        pool = list(by_source.get(source, []))

        rng.shuffle(pool)

        selected.extend(pool[:target])

    if len(selected) < target_size:
        selected_ids = {
            (
                row.get("source_dataset"),
                row.get("source_split"),
                row.get("source_index"),
                row.get("prompt"),
            )
            for row in selected
        }

        leftovers = [
            row
            for row in rows
            if (
                row.get("source_dataset"),
                row.get("source_split"),
                row.get("source_index"),
                row.get("prompt"),
            ) not in selected_ids
        ]

        rng.shuffle(leftovers)

        selected.extend(
            leftovers[:target_size - len(selected)]
        )

    rng.shuffle(selected)

    return selected[:target_size]


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def format_violation_answer(category_name):
    reason = (
        f"The request matches the EU AI Act category "
        f"'{category_name}'."
    )

    return (
        f"VIOLATION\n"
        f"Category: {category_name}\n"
        f"Reason: {reason}"
    ), reason


def format_safe_answer():
    reason = (
        "The request does not match a prohibited practice in the EU AI Act "
        "taxonomy used by this guard model."
    )

    return (
        f"SAFE\n"
        f"Reason: {reason}"
    ), reason


def format_edge_case_answer(category_name):
    reason = (
        "The request is close to an EU AI Act prohibited-practice category, "
        "but the available context is insufficient for a confident SAFE or "
        "VIOLATION determination."
    )
    return (
        f"EDGE_CASE\n"
        f"Closest Category: {category_name}\n"
        f"Reason: {reason}"
    ), reason


def format_row(row, category_names, category_groups):
    label = row.get("label")
    category_id = row.get("category_id")

    if label != "VIOLATION":
        predicted = row.get("predicted_label")
        if predicted == "VIOLATION":
            label = "VIOLATION"
            category_id = row["predicted_category_id"]
        elif predicted == "EDGE_CASE":
            label = "EDGE_CASE"
            category_id = row["predicted_category_id"]
        else:
            label = "SAFE"
            category_id = None

    if label == "VIOLATION" and category_id is None:
        category_id = (
            row.get("category_id")
            or row.get("predicted_category_id")
        )

    if label == "VIOLATION":
        category_name = category_names[category_id]
        answer, reason = format_violation_answer(category_name)
        eu_category = category_name
        group = category_groups.get(category_id)
    elif label == "EDGE_CASE":
        category_name = category_names[category_id]
        answer, reason = format_edge_case_answer(category_name)
        eu_category = category_name
        group = category_groups.get(category_id)
    else:
        answer, reason = format_safe_answer()
        eu_category = "SAFE"
        group = None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": answer},
        ],
        "label": label,
        "eu_ai_act_category": eu_category,
        "group": group,
        "reason": reason,
        "source_dataset": row.get("source_dataset", "UNKNOWN"),
        "source_split": row.get("source_split", "unknown"),
        "source_index": row.get("source_index"),
        "confidence": row.get(
            "confidence",
            "SYNTHETIC" if row.get("source_dataset") == "EU_SYNTHETIC" else "UNKNOWN",
        ),
        "similarity": round(float(row.get("best_score", 1.0)), 6),
        "score_margin": round(float(row.get("score_margin", 1.0)), 6),
    }


# ---------------------------------------------------------------------
# Statistics / writing
# ---------------------------------------------------------------------

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_stats(rows):
    return {
        "total": len(rows),
        "labels": dict(
            Counter(row["label"] for row in rows)
        ),
        "categories": dict(
            Counter(row["eu_ai_act_category"] for row in rows)
        ),
        "sources": dict(
            Counter(row["source_dataset"] for row in rows)
        ),
        "confidence": dict(
            Counter(row.get("confidence") for row in rows)
        ),
        "source_by_label": {
            label: dict(
                Counter(
                    row.get("source_dataset")
                    for row in rows
                    if row.get("label") == label
                )
            )
            for label in ("SAFE", "VIOLATION", "EDGE_CASE")
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EU-Guard Mixed Dataset Builder V5 (REAL + SYNTHETIC QUOTA FILL)"
    )

    parser.add_argument(
        "--eu-reference",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="eu_guard_mixed_v5_5k.jsonl",
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--candidate-limit-per-source",
        type=int,
        default=25000,
    )

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--anchors-per-category",
        type=int,
        default=100,
        help="Maximum synthetic prompt anchors per EU category.",
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.42,
    )

    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.06,
        help="Minimum top-1 vs top-2 category margin for a MEDIUM match to count as VIOLATION.",
    )

    parser.add_argument(
        "--edge-low-threshold",
        type=float,
        default=0.24,
        help="Lower similarity boundary for EDGE_CASE candidates.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--violation-balance",
        type=float,
        default=0.45,
        help="Fraction of final dataset labeled VIOLATION.",
    )

    parser.add_argument(
        "--edge-case-balance",
        type=float,
        default=0.10,
        help="Fraction of final dataset labeled EDGE_CASE. Default 0.10.",
    )

    parser.add_argument(
        "--high-real-ratio",
        type=float,
        default=0.40,
        help="Preferred fraction of each violation quota from HIGH real matches.",
    )

    parser.add_argument(
        "--medium-real-ratio",
        type=float,
        default=0.30,
        help="Preferred fraction of each violation quota from MEDIUM real matches.",
    )

    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.30,
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

    if not 0 < args.violation_balance < 1:
        raise ValueError("--violation-balance must be between 0 and 1.")
    if not 0 <= args.edge_case_balance < 1:
        raise ValueError("--edge-case-balance must be between 0 and 1.")
    if args.violation_balance + args.edge_case_balance >= 1:
        raise ValueError("--violation-balance + --edge-case-balance must be < 1.")

    if not 0 <= args.medium_threshold <= args.high_threshold:
        raise ValueError(
            "--medium-threshold must be <= --high-threshold."
        )
    if not 0 <= args.edge_low_threshold <= args.medium_threshold:
        raise ValueError("--edge-low-threshold must be <= --medium-threshold.")
    if args.margin_threshold < 0:
        raise ValueError("--margin-threshold must be >= 0.")

    if args.high_real_ratio + args.medium_real_ratio > 1:
        raise ValueError(
            "--high-real-ratio + --medium-real-ratio must be <= 1."
        )

    rng = random.Random(args.seed)

    print("=" * 72)
    print("EU-GUARD MIXED DATASET BUILDER V5 (REAL + SYNTHETIC QUOTA FILL)")
    print("=" * 72)

    (
        category_anchor_texts,
        category_names,
        category_groups,
        synthetic_rows,
    ) = load_eu_reference(
        args.eu_reference,
        args.anchors_per_category,
        args.seed,
    )

    print(
        f"\nEU synthetic prompts available: {len(synthetic_rows)}"
    )

    print(
        f"\nLoading embedding model: {args.embedding_model}"
    )

    model = SentenceTransformer(
        args.embedding_model
    )

    hh_rows = load_hh(
        args.candidate_limit_per_source,
        args.seed,
    )

    print(
        f"HH-RLHF prompts extracted: {len(hh_rows)}"
    )

    beaver_rows = load_beaver(
        args.candidate_limit_per_source,
        args.seed,
    )

    print(
        f"BeaverTails prompts extracted: {len(beaver_rows)}"
    )

    real_rows = deduplicate_rows(
        hh_rows + beaver_rows
    )

    print(
        f"\nUnique real candidate prompts: {len(real_rows)}"
    )

    (
        anchor_embeddings,
        category_to_indices,
    ) = build_category_anchor_embeddings(
        model,
        category_anchor_texts,
    )

    classified = classify_candidates(
        model=model,
        rows=real_rows,
        anchor_embeddings=anchor_embeddings,
        category_to_indices=category_to_indices,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
        margin_threshold=args.margin_threshold,
        edge_low_threshold=args.edge_low_threshold,
        top_k=args.top_k,
    )

    confidence_stats = Counter(
        row["confidence"]
        for row in classified
    )

    print("\nCandidate classification:")
    for key in ("HIGH", "MEDIUM", "BOUNDARY", "LOW"):
        print(
            f"  {key}: {confidence_stats.get(key, 0)}"
        )

    violation_target = round(
        args.target_size * args.violation_balance
    )
    edge_target = round(
        args.target_size * args.edge_case_balance
    )
    safe_target = args.target_size - violation_target - edge_target

    print(
        f"\nTarget distribution: "
        f"VIOLATION={violation_target}, EDGE_CASE={edge_target}, SAFE={safe_target}"
    )

    # ------------------------------------------------------------------
    # VIOLATION QUOTA FILL
    # 1) take all available real VIOLATION matches, category-balanced
    # 2) fill only each remaining category deficit from EU_SYNTHETIC
    # ------------------------------------------------------------------
    category_ids = sorted(
        category_names,
        key=lambda x: int(x[1:]),
    )
    base_quota, remainder = divmod(
        violation_target,
        len(category_ids),
    )
    quotas = {
        cid: base_quota + (1 if i < remainder else 0)
        for i, cid in enumerate(category_ids)
    }

    real_by_category = defaultdict(list)
    for row in classified:
        if row.get("predicted_label") == "VIOLATION":
            cid = row.get("predicted_category_id")
            if cid in quotas:
                real_by_category[cid].append(row)

    violations = []
    selected_real_keys = set()
    selection_report = {}

    for cid in category_ids:
        pool = list(real_by_category.get(cid, []))
        rng.shuffle(pool)
        pool.sort(
            key=lambda row: (
                float(row.get("best_score", 0.0)),
                float(row.get("score_margin", 0.0)),
            ),
            reverse=True,
        )
        take = pool[:quotas[cid]]

        for row in take:
            selected_real_keys.add(normalize_for_dedupe(row.get("prompt", "")))
            copied = dict(row)
            copied["label"] = "VIOLATION"
            copied["category_id"] = cid
            violations.append(copied)

        selection_report[cid] = {
            "quota": quotas[cid],
            "real": len(take),
            "synthetic": 0,
            "selected": len(take),
        }

    # Synthetic fill by category. Deduplicate against selected real prompts.
    synthetic_by_category = defaultdict(list)
    seen_synthetic = set(selected_real_keys)

    for row in synthetic_rows:
        cid = row.get("category_id")
        prompt_key = normalize_for_dedupe(row.get("prompt", ""))

        if (
            cid in quotas
            and prompt_key
            and prompt_key not in seen_synthetic
        ):
            seen_synthetic.add(prompt_key)
            copied = dict(row)
            copied["label"] = "VIOLATION"
            copied["category_id"] = cid
            copied["best_score"] = 1.0
            copied["score_margin"] = 1.0
            copied["confidence"] = "SYNTHETIC"
            synthetic_by_category[cid].append(copied)

    for cid in category_ids:
        deficit = quotas[cid] - selection_report[cid]["real"]
        if deficit <= 0:
            continue

        pool = list(synthetic_by_category.get(cid, []))
        rng.shuffle(pool)

        if len(pool) < deficit:
            raise RuntimeError(
                f"Not enough EU synthetic VIOLATION prompts for {cid}: "
                f"need {deficit}, found {len(pool)}. "
                "Increase the EU synthetic reference dataset."
            )

        take = pool[:deficit]
        violations.extend(take)

        selection_report[cid]["synthetic"] = len(take)
        selection_report[cid]["selected"] += len(take)

    if len(violations) != violation_target:
        raise RuntimeError(
            f"VIOLATION quota failed: expected {violation_target}, "
            f"got {len(violations)}."
        )

    print("\nVIOLATION QUOTA FILL:")
    print(f"  Target: {violation_target}")
    print(
        f"  Real selected: "
        f"{sum(x['real'] for x in selection_report.values())}"
    )
    print(
        f"  Synthetic fill: "
        f"{sum(x['synthetic'] for x in selection_report.values())}"
    )

    for cid in category_ids:
        report = selection_report[cid]
        print(
            f"  {cid} | "
            f"quota={report['quota']} | "
            f"real={report['real']} | "
            f"synthetic={report['synthetic']} | "
            f"selected={report['selected']}"
        )

    edge_rows = select_edge_cases(
        classified_rows=classified,
        edge_target=edge_target,
        rng=rng,
    )

    if len(edge_rows) < edge_target:
        print(
            f"WARNING: Only {len(edge_rows)} genuine EDGE_CASE examples were available, "
            f"below target {edge_target}. The final dataset may be smaller unless thresholds are adjusted."
        )

    safe_rows = select_safe(
        classified_rows=classified,
        safe_target=safe_target,
        hard_negative_fraction=args.hard_negative_fraction,
        rng=rng,
    )

    # If real safe rows are insufficient, we do not silently relabel synthetic
    # violations as SAFE.
    if len(safe_rows) < safe_target:
        raise RuntimeError(
            f"Only {len(safe_rows)} SAFE examples available, "
            f"but {safe_target} are required."
        )

    combined = violations + edge_rows + safe_rows

    # Deduplicate one more time before final formatting.
    combined = deduplicate_rows(combined)

    if len(combined) < args.target_size:
        print(
            f"\nWARNING: deduplication reduced final pool to {len(combined)} "
            f"examples, below requested target {args.target_size}."
        )

    rng.shuffle(combined)

    final_rows = [
        format_row(
            row,
            category_names,
            category_groups,
        )
        for row in combined[:args.target_size]
    ]

    # Final shuffle.
    rng.shuffle(final_rows)

    test_size = max(
        1,
        round(
            len(final_rows)
            * args.test_fraction
        ),
    )

    test_rows = final_rows[:test_size]
    train_rows = final_rows[test_size:]

    output_path = Path(args.output)

    train_path = output_path.with_name(
        output_path.stem + "_train.jsonl"
    )

    test_path = output_path.with_name(
        output_path.stem + "_test.jsonl"
    )

    stats_path = output_path.with_name(
        output_path.stem + "_stats.json"
    )

    write_jsonl(
        output_path,
        final_rows,
    )

    write_jsonl(
        train_path,
        train_rows,
    )

    write_jsonl(
        test_path,
        test_rows,
    )

    stats = {
        "all": build_stats(final_rows),
        "train": build_stats(train_rows),
        "test": build_stats(test_rows),
        "selection_report": selection_report,
        "parameters": vars(args),
    }

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            stats,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"All:   {output_path} ({len(final_rows)})"
    )

    print(
        f"Train: {train_path} ({len(train_rows)})"
    )

    print(
        f"Test:  {test_path} ({len(test_rows)})"
    )

    print(
        f"Stats: {stats_path}"
    )

    print("\nFINAL DATASET STATS:")

    print(
        json.dumps(
            build_stats(final_rows),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
