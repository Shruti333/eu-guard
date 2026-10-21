#!/usr/bin/env python3

"""

EU-Guard Qwen SFT + DPO Synthetic Dataset Generator

Pipeline:

    EU AI Act meta-prompts

              |

              v

       Generation LLM

              |

              v

      Structured examples

              |

              v

       Validation/filtering

              |

              +------------------+

              |                  |

              v                  v

         Qwen SFT            Qwen DPO

              |                  |

              v                  v

      train/val/test       train/val/test

IMPORTANT:

- Input meta-prompts use the "meta_prompt" field.

- Generation model generates examples.

- Generation model is NOT asked to behave as the final classifier.

- Final SFT assistant output is strict JSON.

- Metadata is stored separately.

- Splitting is scenario/prompt based to reduce multilingual leakage.

Compatible with OpenAI-compatible APIs:

    OpenRouter

    vLLM

    Ollama-compatible servers

    Unsloth API

    other /v1/chat/completions servers

"""

from __future__ import annotations

import argparse

import hashlib

import json

import os

import random

import re

import time

from collections import Counter, defaultdict

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed



# ============================================================================

# CONSTANTS

# ============================================================================

VALID_LEVELS = {

    "VIOLATION",

    "EDGE_CASE",

    "COMPLIANT",

    "SAFE",

}

VALID_LABELS = {

    "VIOLATION",

    "EDGE_CASE",

    "SAFE",

}

VALID_CATEGORIES = {

    "G1",

    "G2",

    "G3",

    "G4",

    "G5",

    "G6",

    "G7",

}

CATEGORY_DESCRIPTIONS = {

    "G1": "Sensitive attribute inference from biometric or other data.",

    "G2": "Emotion recognition in workplace or educational settings.",

    "G3": "Biometric categorisation using sensitive characteristics.",

    "G4": "Social scoring of natural persons.",

    "G5": "Predictive policing or criminal risk assessment based on profiling.",

    "G6": "Real-time remote biometric identification in publicly accessible spaces.",

    "G7": "Other EU AI Act prohibited practices represented by the dataset taxonomy.",

}



# ============================================================================

# FINAL QWEN CLASSIFIER SYSTEM PROMPT

# ============================================================================

QWEN_CLASSIFIER_SYSTEM_PROMPT = """You are an EU AI Act safety classification model.

Your task is to classify the user's request according to the EU AI Act

prohibition taxonomy used in this dataset.

Classify the user's request as exactly one of:

SAFE

EDGE_CASE

VIOLATION

For VIOLATION, identify the relevant prohibition group.

For EDGE_CASE, identify the potentially relevant prohibition group when one

exists.

For SAFE, category_id must be null.

Return ONLY valid JSON.

Required format:

{

  "label": "SAFE | EDGE_CASE | VIOLATION",

  "category_id": "G1 | G2 | G3 | G4 | G5 | G6 | G7 | null"

}

Do not output markdown.

Do not output explanations.

Do not output any text outside the JSON object.

"""



# ============================================================================

# GENERATION SYSTEM PROMPT

# ============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a dataset-generation expert creating

high-quality examples for an EU AI Act safety classifier.

Your job is NOT to classify the input meta-prompt.

Your job is to GENERATE realistic user requests that exercise the specific

EU AI Act prohibition described by the meta-prompt.

The generated examples will later be used to fine-tune a Qwen safety

classification model.

QUALITY REQUIREMENTS

====================

1. Follow the EU AI Act category specified by the meta-prompt exactly.

2. Generate realistic user requests that a person might actually send to an

AI assistant.

3. Do NOT generate generic conversational text.

4. Do NOT use random unrelated harmless questions.

5. Do NOT copy the meta-prompt.

6. Each example must contain enough context for the safety category to be

understood.

7. Avoid extremely short examples such as:

   "Is this allowed?"

   "What do you think?"

   "Yes, do it."

   unless the surrounding context makes the classification genuinely clear.

8. Prefer concrete scenarios involving:

   - an AI system

   - a person or group

   - the intended use

   - relevant data

   - the requested action

9. Vary the surface form substantially.

10. Use different:

    - industries

    - organizations

    - user motivations

    - sentence structures

    - levels of directness

    - realistic contexts

11. Do not make every example explicitly mention the EU AI Act.

12. Do not make every example start with "Can you".

13. Do not inject the category name into the user request unless natural.

14. Do not create obviously artificial benchmark language.

15. For VIOLATION examples, the prohibited behavior should actually be

    requested or meaningfully facilitated.

16. For SAFE examples, create genuinely safe requests that may be related

    to the same domain but do not perform the prohibited practice.

17. For EDGE_CASE examples, create realistic borderline situations where

    the relevant prohibition may depend on context, implementation,

    purpose, safeguards, or deployment setting.

18. Never label an unrelated generic request as a violation simply because

    it contains words associated with the category.

19. Each generated example must be independently understandable.

20. Do not include the final classifier answer in the user prompt.

OUTPUT FORMAT

=============

Return ONLY valid JSON.

Use exactly this structure:

{

  "examples": [

    {

      "prompt": "...",

      "label": "VIOLATION",

      "category_id": "G1",

      "scenario_type": "direct_request",

      "context": "...",

      "target_behavior": "...",

      "rationale": "..."

    }

  ]

}

Allowed label values:

- VIOLATION

- EDGE_CASE

- SAFE

Allowed category_id values:

- G1

- G2

- G3

- G4

- G5

- G6

- G7

- null

For SAFE examples, category_id MUST be null.

For EDGE_CASE examples, category_id should identify the potentially relevant

group when there is one.

The "prompt" field is the actual user message that will be used for training.

The "context", "target_behavior", and "rationale" fields are metadata for

dataset auditing and must NOT be inserted into the final user message.

Generate diverse, high-quality examples rather than repetitive variations.

"""



# ============================================================================

# OPTIONAL VALIDATOR PROMPT

# ============================================================================

VALIDATOR_SYSTEM_PROMPT = """You are an independent quality-control evaluator

for an EU AI Act safety-classification dataset.

Evaluate ONE generated user request.

You must determine whether:

1. the requested label is supported by the request,

2. the category is supported by the request,

3. the request is sufficiently realistic and specific,

4. the example is not generic or unrelated,

5. the example does not merely contain category-related keywords without

   actually describing the prohibited behavior.

Return ONLY valid JSON:

{

  "valid": true,

  "label_correct": true,

  "category_correct": true,

  "realistic": true,

  "specific": true,

  "reason": "short explanation"

}

If the example is wrong, set "valid" to false.

"""



# ============================================================================

# UTILITY FUNCTIONS

# ============================================================================

def clean_text(value: Any) -> str:

    if value is None:

        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")

    text = text.replace("\r", "\n")

    # Remove accidental markdown fences.

    text = re.sub(r"^\`\`\`(?:json)?\s*", "", text.strip(), flags=re.I)

    text = re.sub(r"\s*\`\`\`$", "", text.strip())

    return text.strip()



def normalize_whitespace(text: str) -> str:

    return re.sub(r"\s+", " ", text).strip()



def hash_text(text: str) -> str:

    return hashlib.sha256(

        normalize_whitespace(text).lower().encode("utf-8")

    ).hexdigest()



def group_to_category_id(group: Any) -> Optional[str]:

    if group is None:

        return None

    value = str(group).strip().upper()

    if value.startswith("G"):

        try:

            number = int(value[1:])

            if 1 <= number <= 7:

                return f"G{number}"

        except ValueError:

            return None

    try:

        number = int(group)

        if 1 <= number <= 7:

            return f"G{number}"

    except (ValueError, TypeError):

        pass

    return None



def normalize_level(level: Any) -> str:

    value = str(level or "").strip().upper()

    if value == "COMPLIANT":

        return "SAFE"

    return value



def expected_label_from_level(level: str) -> str:

    level = normalize_level(level)

    if level == "VIOLATION":

        return "VIOLATION"

    if level == "EDGE_CASE":

        return "EDGE_CASE"

    if level in {"COMPLIANT", "SAFE"}:

        return "SAFE"

    raise ValueError(f"Unknown level: {level}")



def safe_json_loads(text: str) -> Optional[Any]:

    text = clean_text(text)

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass

    # Find first JSON object.

    start = text.find("{")

    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[start:end + 1]

        try:

            return json.loads(candidate)

        except json.JSONDecodeError:

            pass

    # Find JSON array.

    start = text.find("[")

    end = text.rfind("]")

    if start >= 0 and end > start:

        candidate = text[start:end + 1]

        try:

            return json.loads(candidate)

        except json.JSONDecodeError:

            pass

    return None



# ============================================================================

# LOAD META PROMPTS

# ============================================================================

def load_meta_prompts(

    path: str,

    max_meta_prompts: Optional[int] = None,

) -> List[Dict[str, Any]]:

    records: List[Dict[str, Any]] = []

    input_path = Path(path)

    if not input_path.exists():

        raise FileNotFoundError(

            f"Meta-prompt file does not exist: {input_path}"

        )

    with input_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:

                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError as e:

                print(

                    f"[WARNING] Invalid JSON at line "

                    f"{line_number}: {e}"

                )

                continue

            if not isinstance(row, dict):

                continue

            meta_prompt = row.get("meta_prompt")

            if not isinstance(meta_prompt, str):

                print(

                    f"[WARNING] Row {line_number}: "

                    f"missing meta_prompt"

                )

                continue

            meta_prompt = clean_text(meta_prompt)

            if not meta_prompt:

                continue

            category_id = group_to_category_id(

                row.get("group")

            )

            if category_id is None:

                print(

                    f"[WARNING] Row {line_number}: "

                    f"invalid group {row.get('group')!r}"

                )

                continue

            level = normalize_level(

                row.get("level")

            )

            if level not in VALID_LEVELS:

                print(

                    f"[WARNING] Row {line_number}: "

                    f"invalid level {level!r}"

                )

                continue

            normalized = dict(row)

            normalized["_meta_prompt"] = meta_prompt

            normalized["_category_id"] = category_id

            normalized["_level"] = level

            records.append(normalized)

            if (

                max_meta_prompts is not None

                and len(records) >= max_meta_prompts

            ):

                break

    return records



# ============================================================================

# OPENAI-COMPATIBLE MODEL CALL

# ============================================================================

def call_model(

    client: OpenAI,

    model: str,

    messages: List[Dict[str, str]],

    temperature: float,

    max_tokens: int,

    retries: int = 3,

) -> str:

    last_error = None

    for attempt in range(1, retries + 1):

        try:

            response = client.chat.completions.create(

                model=model,

                messages=messages,

                temperature=temperature,

                max_tokens=max_tokens,

                response_format={"type": "json_object"},

            )

            if not response.choices:

                raise RuntimeError(

                    "Model returned no choices."

                )

            content = response.choices[0].message.content

            if content is None:

                raise RuntimeError(

                    "Model returned empty content."

                )

            return content.strip()

        except Exception as e:

            last_error = e

            print(

                f"[WARNING] Model call failed "

                f"(attempt {attempt}/{retries}): {e}"

            )

            if attempt < retries:

                time.sleep(

                    min(2 ** (attempt - 1), 10)

                )

    raise RuntimeError(

        f"Model call failed after {retries} attempts: "

        f"{last_error}"

    )



# ============================================================================

# GENERATION

# ============================================================================

def build_generation_user_prompt(

    record: Dict[str, Any],

    examples_per_meta: int,

) -> str:

    category_id = record["_category_id"]

    level = record["_level"]

    category_description = CATEGORY_DESCRIPTIONS.get(

        category_id,

        "Use the exact category definition contained in the meta-prompt."

    )

    return f"""

Generate {examples_per_meta} diverse synthetic user requests.

SOURCE DATASET CATEGORY:

{category_id}

CATEGORY DESCRIPTION:

{category_description}

SOURCE LABEL:

{level}

IMPORTANT:

The source meta-prompt below is authoritative for the intended scenario.

SOURCE META-PROMPT:

-------------------

{record["_meta_prompt"]}

-------------------

Generate exactly {examples_per_meta} examples.

The examples must be meaningfully different scenarios, not superficial

rewrites of the same sentence.

For this batch, make the examples realistic enough that a human evaluator

could understand why each example belongs to the requested category.

Remember:

- Do not generate unrelated generic questions.

- Do not rely only on category keywords.

- Do not invent a violation merely because the prompt mentions AI,

  biometrics, policing, employment, education, or personal data.

- Follow the exact prohibition described by the source meta-prompt.

Return ONLY the JSON object requested by the system prompt.

""".strip()



def generate_examples_for_record(

    client: OpenAI,

    model: str,

    record: Dict[str, Any],

    examples_per_meta: int,

    temperature: float,

    max_tokens: int,

) -> List[Dict[str, Any]]:

    user_prompt = build_generation_user_prompt(

        record,

        examples_per_meta,

    )

    raw = call_model(

        client=client,

        model=model,

        messages=[

            {

                "role": "system",

                "content": GENERATOR_SYSTEM_PROMPT,

            },

            {

                "role": "user",

                "content": user_prompt,

            },

        ],

        temperature=temperature,

        max_tokens=max_tokens,

    )

    parsed = safe_json_loads(raw)

    if parsed is None:

        print(

            "[WARNING] Generator returned invalid JSON"

        )

        return []

    if isinstance(parsed, dict):

        examples = parsed.get("examples", [])

    elif isinstance(parsed, list):

        examples = parsed

    else:

        return []

    if not isinstance(examples, list):

        return []

    results = []

    expected_label = expected_label_from_level(

        record["_level"]

    )

    expected_category = record["_category_id"]

    for index, example in enumerate(examples):

        if not isinstance(example, dict):

            continue

        prompt = clean_text(

            example.get("prompt")

        )

        if not prompt:

            continue

        label = str(

            example.get("label", "")

        ).strip().upper()

        category_id = example.get(

            "category_id"

        )

        if category_id is not None:

            category_id = str(

                category_id

            ).strip().upper()

        # ------------------------------------------------------------

        # Enforce source category / label.

        # We do not trust the generator to silently change the target.

        # ------------------------------------------------------------

        if label != expected_label:

            print(

                f"[REJECT] Wrong label: "

                f"expected={expected_label}, "

                f"got={label}"

            )

            continue

        if expected_label == "SAFE":

            category_id = None

        elif category_id != expected_category:

            print(

                f"[REJECT] Wrong category: "

                f"expected={expected_category}, "

                f"got={category_id}"

            )

            continue

        # ------------------------------------------------------------

        # Basic quality checks

        # ------------------------------------------------------------

        words = prompt.split()

        if len(words) < 8:

            print(

                f"[REJECT] Too short: {prompt}"

            )

            continue

        if len(words) > 300:

            print(

                f"[REJECT] Too long."

            )

            continue

        # Reject obvious meta/instruction leakage.

        lowered = prompt.lower()

        forbidden_meta_phrases = [

            "classify this prompt",

            "eu ai act category",

            "category_id",

            "output only valid json",

            "source meta-prompt",

            "dataset generation",

            "training example",

        ]

        if any(

            phrase in lowered

            for phrase in forbidden_meta_phrases

        ):

            print(

                "[REJECT] Meta-prompt leakage."

            )

            continue

        results.append(

            {

                "prompt": prompt,

                "label": label,

                "category_id": category_id,

                "scenario_type": clean_text(

                    example.get(

                        "scenario_type",

                        "unknown"

                    )

                ),

                "context": clean_text(

                    example.get("context", "")

                ),

                "target_behavior": clean_text(

                    example.get(

                        "target_behavior",

                        ""

                    )

                ),

                "rationale": clean_text(

                    example.get(

                        "rationale",

                        ""

                    )

                ),

                "meta_id": record.get(

                    "id",

                    f"meta_{hash_text(record['_meta_prompt'])[:12]}"

                ),

                "source_group": record["_category_id"],

                "source_level": record["_level"],

                "language_code": record.get(

                    "language_code"

                ),

                "language_name": record.get(

                    "language_name"

                ),

                "augmentation": record.get(

                    "augmentation"

                ),

                "variant_index": record.get(

                    "variant_index"

                ),

                "example_index": index,

            }

        )

    return results



# ============================================================================

# VALIDATION

# ============================================================================

def validate_example_locally(

    example: Dict[str, Any]

) -> Tuple[bool, str]:

    prompt = clean_text(

        example.get("prompt")

    )

    if not prompt:

        return False, "empty_prompt"

    label = example.get("label")

    category_id = example.get(

        "category_id"

    )

    if label not in VALID_LABELS:

        return False, "invalid_label"

    if label == "SAFE":

        if category_id is not None:

            return False, "safe_has_category"

    else:

        if category_id not in VALID_CATEGORIES:

            return False, "missing_or_invalid_category"

    words = prompt.split()

    if len(words) < 8:

        return False, "too_short"

    if len(words) > 300:

        return False, "too_long"

    return True, "ok"



def validate_with_model(

    client: OpenAI,

    validator_model: str,

    example: Dict[str, Any],

    temperature: float = 0.0,

    max_tokens: int = 300,

) -> Tuple[bool, Dict[str, Any]]:

    prompt = example["prompt"]

    label = example["label"]

    category = example["category_id"]

    user_message = f"""

Evaluate this generated example.

EXPECTED LABEL:

{label}

EXPECTED CATEGORY:

{category}

USER REQUEST:

{prompt}

""".strip()

    raw = call_model(

        client=client,

        model=validator_model,

        messages=[

            {

                "role": "system",

                "content": VALIDATOR_SYSTEM_PROMPT,

            },

            {

                "role": "user",

                "content": user_message,

            },

        ],

        temperature=temperature,

        max_tokens=max_tokens,

    )

    print("\n[DEBUG] RAW MODEL RESPONSE:")

    print(raw[:10080])

    print("[DEBUG] END RAW RESPONSE\n")

    parsed = safe_json_loads(raw)

    if not isinstance(parsed, dict):

        return False, {

            "valid": False,

            "reason": "validator_invalid_json",

        }

    valid = bool(

        parsed.get("valid", False)

    )

    return valid, parsed



# ============================================================================

# DEDUPLICATION

# ============================================================================

def deduplicate_examples(

    examples: List[Dict[str, Any]]

) -> List[Dict[str, Any]]:

    seen = set()

    output = []

    for example in examples:

        key = hash_text(

            example["prompt"]

        )

        if key in seen:

            continue

        seen.add(key)

        example["prompt_hash"] = key

        output.append(example)

    return output



# ============================================================================

# PROMPT / SCENARIO GROUP ID

# ============================================================================

def assign_prompt_ids(

    examples: List[Dict[str, Any]]

) -> None:

    for example in examples:

        # The underlying source meta prompt is the primary scenario

        # grouping key. This prevents multilingual variants from being

        # randomly distributed across train and test.

        base = example.get(

            "meta_id",

            "unknown"

        )

        augmentation = example.get(

            "augmentation",

            "NONE"

        )

        variant = example.get(

            "variant_index",

            0

        )

        scenario_key = (

            f"{base}|"

            f"{augmentation}|"

            f"{variant}|"

            f"{example.get('example_index', 0)}"

        )

        example["prompt_id"] = (

            "p_"

            + hashlib.sha256(

                scenario_key.encode("utf-8")

            ).hexdigest()[:16]

        )



# ============================================================================

# QWEN SFT FORMAT

# ============================================================================

def build_qwen_sft_example(

    example: Dict[str, Any]

) -> Dict[str, Any]:

    assistant_output = {

        "label": example["label"],

        "category_id": example["category_id"],

    }

    return {

        "messages": [

            {

                "role": "system",

                "content": QWEN_CLASSIFIER_SYSTEM_PROMPT,

            },

            {

                "role": "user",

                "content": example["prompt"],

            },

            {

                "role": "assistant",

                "content": json.dumps(

                    assistant_output,

                    ensure_ascii=False,

                    separators=(",", ":"),

                ),

            },

        ]

    }



# ============================================================================

# DPO FORMAT

# ============================================================================

def build_dpo_pair(

    example: Dict[str, Any]

) -> Optional[Dict[str, Any]]:

    """

    Creates a DPO pair where the chosen answer is the correct classifier

    output and rejected answer is a plausible incorrect classification.

    This is deliberately NOT a random arbitrary answer.

    The rejected answer is selected from a realistic nearby decision boundary.

    """

    label = example["label"]

    category = example["category_id"]

    chosen = json.dumps(

        {

            "label": label,

            "category_id": category,

        },

        ensure_ascii=False,

        separators=(",", ":"),

    )

    # ------------------------------------------------------------

    # Construct a plausible hard negative.

    # ------------------------------------------------------------

    if label == "VIOLATION":

        rejected = json.dumps(

            {

                "label": "SAFE",

                "category_id": None,

            },

            ensure_ascii=False,

            separators=(",", ":"),

        )

    elif label == "SAFE":

        rejected = json.dumps(

            {

                "label": "VIOLATION",

                "category_id": (

                    category

                    if category

                    else "G1"

                ),

            },

            ensure_ascii=False,

            separators=(",", ":"),

        )

    else:

        # EDGE_CASE:

        # The most useful contrast is often SAFE or VIOLATION.

        rejected = json.dumps(

            {

                "label": "SAFE",

                "category_id": None,

            },

            ensure_ascii=False,

            separators=(",", ":"),

        )

    return {

        "prompt": example["prompt"],

        "chosen": chosen,

        "rejected": rejected,

    }



# ============================================================================

# GROUP-AWARE SPLIT

# ============================================================================

def split_by_prompt_group(

    examples: List[Dict[str, Any]],

    train_ratio: float,

    val_ratio: float,

    seed: int,

) -> Tuple[

    List[Dict[str, Any]],

    List[Dict[str, Any]],

    List[Dict[str, Any]],

]:

    rng = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for example in examples:

        groups[

            example["prompt_id"]

        ].append(example)

    group_items = list(

        groups.items()

    )

    rng.shuffle(group_items)

    total_groups = len(group_items)

    train_count = int(

        total_groups * train_ratio

    )

    val_count = int(

        total_groups * val_ratio

    )

    train_groups = group_items[

        :train_count

    ]

    val_groups = group_items[

        train_count:

        train_count + val_count

    ]

    test_groups = group_items[

        train_count + val_count:

    ]

    def flatten(

        items: List[

            Tuple[

                str,

                List[Dict[str, Any]]

            ]

        ]

    ) -> List[Dict[str, Any]]:

        output = []

        for _, records in items:

            output.extend(records)

        return output

    return (

        flatten(train_groups),

        flatten(val_groups),

        flatten(test_groups),

    )



# ============================================================================

# WRITERS

# ============================================================================

def write_jsonl(

    path: Path,

    rows: Iterable[Dict[str, Any]],

) -> None:

    with path.open(

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



def write_sft_jsonl(

    path: Path,

    examples: List[Dict[str, Any]],

) -> None:

    write_jsonl(

        path,

        (

            build_qwen_sft_example(

                example

            )

            for example in examples

        ),

    )



def write_dpo_jsonl(

    path: Path,

    examples: List[Dict[str, Any]],

) -> None:

    rows = []

    for example in examples:

        pair = build_dpo_pair(

            example

        )

        if pair is not None:

            rows.append(pair)

    write_jsonl(

        path,

        rows,

    )



# ============================================================================

# STATISTICS

# ============================================================================

def make_stats(

    examples: List[Dict[str, Any]],

) -> Dict[str, Any]:

    label_counts = Counter(

        x["label"]

        for x in examples

    )

    category_counts = Counter(

        x["category_id"]

        for x in examples

        if x["category_id"]

    )

    language_counts = Counter(

        x.get("language_code")

        for x in examples

        if x.get("language_code")

    )

    return {

        "total_examples": len(examples),

        "labels": dict(label_counts),

        "categories": dict(category_counts),

        "languages": dict(language_counts),

        "unique_prompts": len(

            {

                x["prompt_id"]

                for x in examples

            }

        ),

    }



# ============================================================================

# MAIN

# ============================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Generate Qwen SFT/DPO dataset "

            "from EU-Guard meta-prompts."

        )

    )

    parser.add_argument(

        "--meta-prompts",

        required=True,

        help="EU AI Act meta-prompt JSONL",

    )

    parser.add_argument(

        "--base-url",

        required=True,

        help="OpenAI-compatible API base URL",

    )

    parser.add_argument(

        "--api-key",

        default=None,

        help="API key. Defaults to OPENAI_API_KEY "

             "or OPENROUTER_API_KEY.",

    )

    parser.add_argument(

        "--model",

        required=True,

        help="Generation model name",

    )

    parser.add_argument(

        "--validator-model",

        default=None,

        help=(

            "Optional independent validator model. "

            "If omitted, validation is local only."

        ),

    )

    parser.add_argument(

        "--out-dir",

        default="./gemini_dataset_20k",

    )

    parser.add_argument(

        "--max-meta-prompts",

        type=int,

        default=None,

    )

    parser.add_argument(

        "--examples-per-meta",

        type=int,

        default=5,

    )

    parser.add_argument(

        "--target-examples",

        type=int,

        default=None,

    )

    parser.add_argument(

        "--temperature",

        type=float,

        default=0.8,

    )

    parser.add_argument(

        "--max-tokens",

        type=int,

        default=4096,

    )

    parser.add_argument(

        "--validator-max-tokens",

        type=int,

        default=300,

    )

    parser.add_argument(

        "--validate-with-model",

        action="store_true",

    )

    parser.add_argument(

        "--train-ratio",

        type=float,

        default=0.90,

    )

    parser.add_argument(

        "--val-ratio",

        type=float,

        default=0.05,

    )

    parser.add_argument(

        "--seed",

        type=int,

        default=42,

    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent API workers. Recommended: 4-8.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint_examples.jsonl in --out-dir.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------

    # API key

    # ------------------------------------------------------------

    api_key = (

    args.api_key

    or os.getenv("OPENROUTER_API_KEY")

    )

    if not api_key:

        raise RuntimeError(

            "No OPENROUTER_API_KEY supplied. "

            "Set your OpenRouter key or use --api-key."

        )

    # ------------------------------------------------------------

    # Output directory

    # ------------------------------------------------------------

    out_dir = Path(

        args.out_dir

    )

    out_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    # ------------------------------------------------------------

    # Client

    # ------------------------------------------------------------



    print(

    "[DEBUG] API key loaded:",

    bool(api_key),

    "prefix:",

    api_key[:12] if api_key else None

    )

    client = OpenAI(

    api_key=api_key,

    base_url=args.base_url.rstrip("/"),

    )



    # ------------------------------------------------------------

    # Load meta prompts

    # ------------------------------------------------------------

    records = load_meta_prompts(

        args.meta_prompts,

        args.max_meta_prompts,

    )

    print(

        f"[INFO] Loaded {len(records)} meta-prompts"

    )

    if not records:

        raise RuntimeError(

            "No valid meta-prompts found."

        )

    # ------------------------------------------------------------
    # Generate (parallel + durable checkpoint/resume)
    # ------------------------------------------------------------

    checkpoint_path = out_dir / "checkpoint_examples.jsonl"
    state_path = out_dir / "checkpoint_state.json"
    checkpoint_lock = threading.Lock()

    all_examples: List[Dict[str, Any]] = []
    completed_meta_ids = set()

    def meta_id_for_record(record):
        return str(record.get(
            "id",
            f"meta_{hash_text(record['_meta_prompt'])[:12]}"
        ))

    def load_checkpoint():
        loaded = []
        completed = set()

        if not args.resume:
            return loaded, completed

        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        print("[WARNING] Ignoring malformed checkpoint line.")
                        continue

                    mid = item.get("meta_id")
                    examples = item.get("examples", [])
                    if mid and isinstance(examples, list):
                        loaded.extend(
                            x for x in examples if isinstance(x, dict)
                        )
                        completed.add(str(mid))

        print(
            f"[RESUME] Loaded {len(loaded)} checkpoint examples "
            f"from {len(completed)} completed meta-prompts."
        )
        return loaded, completed

    checkpoint_examples, completed_meta_ids = load_checkpoint()
    all_examples.extend(checkpoint_examples)

    remaining_records = [
        r for r in records
        if meta_id_for_record(r) not in completed_meta_ids
    ]

    print(
        f"[INFO] Remaining meta-prompts: "
        f"{len(remaining_records)}/{len(records)}"
    )
    print(f"[INFO] Parallel workers: {max(1, args.workers)}")

    def process_one(record):
        mid = meta_id_for_record(record)
        print(
            f"\n[INFO] Processing {mid} "
            f"category={record['_category_id']} "
            f"level={record['_level']} "
            f"language={record.get('language_code')}"
        )

        worker_client = OpenAI(
            api_key=api_key,
            base_url=args.base_url.rstrip("/"),
        )

        try:
            examples = generate_examples_for_record(
                client=worker_client,
                model=args.model,
                record=record,
                examples_per_meta=args.examples_per_meta,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"[ERROR] Generation failed for {mid}: {e}")
            return mid, [], False

        print(f"[INFO] {mid}: Generated {len(examples)} candidate examples")

        accepted = []
        for example in examples:
            valid, reason = validate_example_locally(example)

            if not valid:
                print(
                    f"[REJECT] {mid} {reason}: "
                    f"{example.get('prompt', '')[:100]}"
                )
                continue

            if args.validate_with_model and args.validator_model:
                try:
                    valid_by_model, report = validate_with_model(
                        client=worker_client,
                        validator_model=args.validator_model,
                        example=example,
                        temperature=0.0,
                        max_tokens=args.validator_max_tokens,
                    )

                    if not valid_by_model:
                        print(
                            f"[REJECT] {mid} Validator: "
                            f"{report.get('reason', '')}"
                        )
                        continue

                    example["validator_report"] = report

                except Exception as e:
                    print(f"[WARNING] Validator failed for {mid}: {e}")
                    continue

            accepted.append(example)

        return mid, accepted, True

    def checkpoint_result(mid, examples):
        with checkpoint_lock:
            with checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"meta_id": mid, "examples": examples},
                        ensure_ascii=False,
                    ) + "\n"
                )
                f.flush()
                os.fsync(f.fileno())

            completed_meta_ids.add(mid)

            state = {
                "version": 1,
                "completed_meta_ids": sorted(completed_meta_ids),
                "completed_count": len(completed_meta_ids),
                "updated_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            }

            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for attempt in range(5):
                try:
                    os.replace(tmp, state_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(1)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_one, record): record
            for record in remaining_records
        }

        for future in as_completed(futures):
            record = futures[future]
            mid = meta_id_for_record(record)

            try:
                mid, examples, success = future.result()
            except Exception as e:
                print(f"[ERROR] Worker crashed for {mid}: {e}")
                continue

            if not success:
                # Not checkpointed; --resume will retry it.
                continue

            checkpoint_result(mid, examples)
            all_examples.extend(examples)

            print(
                f"[CHECKPOINT] {len(completed_meta_ids)}/{len(records)} "
                f"meta-prompts saved | {len(all_examples)} examples loaded"
            )

    print(
        f"\n[INFO] Generation phase complete. "
        f"Checkpointed {len(completed_meta_ids)}/{len(records)} meta-prompts."
    )

    # ------------------------------------------------------------

    # Deduplicate

    # ------------------------------------------------------------

    print(

        f"\n[INFO] Before deduplication: "

        f"{len(all_examples)}"

    )

    all_examples = deduplicate_examples(

        all_examples

    )

    # ------------------------------------------------------------

    # Target limit

    # ------------------------------------------------------------

    if args.target_examples:

        all_examples = all_examples[

            :args.target_examples

        ]

    # ------------------------------------------------------------

    # Assign prompt IDs

    # ------------------------------------------------------------

    assign_prompt_ids(

        all_examples

    )

    print(

        f"[INFO] Final examples: "

        f"{len(all_examples)}"

    )

    # ------------------------------------------------------------

    # Split

    # ------------------------------------------------------------

    train_examples, val_examples, test_examples = (

        split_by_prompt_group(

            all_examples,

            train_ratio=args.train_ratio,

            val_ratio=args.val_ratio,

            seed=args.seed,

        )

    )

    print(

        f"[INFO] Train: {len(train_examples)}"

    )

    print(

        f"[INFO] Validation: {len(val_examples)}"

    )

    print(

        f"[INFO] Test: {len(test_examples)}"

    )

    # ------------------------------------------------------------

    # Write SFT files

    # ------------------------------------------------------------

    write_sft_jsonl(

        out_dir / "qwen_train.jsonl",

        train_examples,

    )

    write_sft_jsonl(

        out_dir / "qwen_val.jsonl",

        val_examples,

    )

    write_sft_jsonl(

        out_dir / "qwen_test.jsonl",

        test_examples,

    )

    write_sft_jsonl(

        out_dir / "qwen_all.jsonl",

        all_examples,

    )

    # ------------------------------------------------------------

    # Write DPO files

    # ------------------------------------------------------------

    write_dpo_jsonl(

        out_dir / "dpo_train.jsonl",

        train_examples,

    )

    write_dpo_jsonl(

        out_dir / "dpo_val.jsonl",

        val_examples,

    )

    write_dpo_jsonl(

        out_dir / "dpo_test.jsonl",

        test_examples,

    )

    # ------------------------------------------------------------

    # Metadata

    # ------------------------------------------------------------

    metadata = []

    for example in all_examples:

        metadata.append(

            {

                k: v

                for k, v in example.items()

                if k != "validator_report"

            }

        )

    write_jsonl(

        out_dir / "metadata.jsonl",

        metadata,

    )

    # ------------------------------------------------------------

    # Stats

    # ------------------------------------------------------------

    stats = make_stats(

        all_examples

    )

    stats.update(

        {

            "model": args.model,

            "validator_model": (

                args.validator_model

                if args.validate_with_model

                else None

            ),

            "examples_per_meta": (

                args.examples_per_meta

            ),

            "train_count": len(

                train_examples

            ),

            "val_count": len(

                val_examples

            ),

            "test_count": len(

                test_examples

            ),

            "format": "Qwen messages SFT",

            "dpo_format": (

                "prompt/chosen/rejected"

            ),

        }

    )

    with (

        out_dir / "stats.json"

    ).open(

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            stats,

            f,

            ensure_ascii=False,

            indent=2,

        )

    # ------------------------------------------------------------

    # Summary

    # ------------------------------------------------------------

    print("\n" + "=" * 70)

    print("DATASET GENERATION COMPLETE")

    print("=" * 70)

    print(

        f"Total:      {len(all_examples)}"

    )

    print(

        f"Train:      {len(train_examples)}"

    )

    print(

        f"Validation: {len(val_examples)}"

    )

    print(

        f"Test:       {len(test_examples)}"

    )

    print("\nFiles:")

    print(

        f"  {out_dir / 'qwen_train.jsonl'}"

    )

    print(

        f"  {out_dir / 'qwen_val.jsonl'}"

    )

    print(

        f"  {out_dir / 'qwen_test.jsonl'}"

    )

    print(

        f"  {out_dir / 'qwen_all.jsonl'}"

    )

    print(

        f"  {out_dir / 'dpo_train.jsonl'}"

    )

    print(

        f"  {out_dir / 'dpo_val.jsonl'}"

    )

    print(

        f"  {out_dir / 'dpo_test.jsonl'}"

    )

    print(

        f"  {out_dir / 'metadata.jsonl'}"

    )

    print(

        f"  {out_dir / 'stats.json'}"

    )

    print("=" * 70)



if __name__ == "__main__":

    main()