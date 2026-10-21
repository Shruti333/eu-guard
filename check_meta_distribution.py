import json
from collections import Counter

INPUT = "eu_ai_act_dataset_v3.jsonl"

rows = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print("=" * 70)
print(f"TOTAL META-PROMPTS: {len(rows)}")
print("=" * 70)

# ------------------------------------------------------------
# GROUP
# ------------------------------------------------------------

groups = Counter(
    f"G{r.get('group')}" if not str(r.get("group", "")).startswith("G")
    else str(r.get("group"))
    for r in rows
)

print("\nGROUP DISTRIBUTION")
for k, v in sorted(groups.items()):
    print(f"{k:5} {v:5}  ({v/len(rows)*100:6.2f}%)")


# ------------------------------------------------------------
# LEVEL
# ------------------------------------------------------------

levels = Counter(r.get("level") for r in rows)

print("\nLEVEL DISTRIBUTION")
for k, v in levels.items():
    print(f"{k:12} {v:5}  ({v/len(rows)*100:6.2f}%)")


# ------------------------------------------------------------
# LANGUAGE
# ------------------------------------------------------------

languages = Counter(r.get("language_code") for r in rows)

print("\nLANGUAGE DISTRIBUTION")
for k, v in sorted(languages.items()):
    print(f"{k:5} {v:5}")


# ------------------------------------------------------------
# PERSONA
# ------------------------------------------------------------

personas = Counter(r.get("persona_tone") for r in rows)

print("\nPERSONA DISTRIBUTION")
for k, v in sorted(personas.items()):
    print(f"{str(k):15} {v:5}")


# ------------------------------------------------------------
# AUGMENTATION
# ------------------------------------------------------------

augmentations = Counter(r.get("augmentation") for r in rows)

print("\nAUGMENTATION DISTRIBUTION")
for k, v in sorted(augmentations.items()):
    print(f"{str(k):20} {v:5}")


# ------------------------------------------------------------
# GROUP × LEVEL
# ------------------------------------------------------------

print("\nGROUP × LEVEL")

combo = Counter(
    (
        f"G{r.get('group')}" if not str(r.get("group", "")).startswith("G")
        else str(r.get("group")),
        r.get("level")
    )
    for r in rows
)

for group in sorted(set(x[0] for x in combo)):
    print(f"\n{group}")
    for level in ["VIOLATION", "EDGE_CASE", "COMPLIANT", "SAFE"]:
        count = combo.get((group, level), 0)
        if count:
            print(f"  {level:12} {count}")


# ------------------------------------------------------------
# GROUP × LANGUAGE
# ------------------------------------------------------------

print("\nGROUP × LANGUAGE")

for group in sorted(set(
    f"G{r.get('group')}" if not str(r.get("group", "")).startswith("G")
    else str(r.get("group"))
    for r in rows
)):
    lang_counts = Counter(
        r.get("language_code")
        for r in rows
        if (
            f"G{r.get('group')}" if not str(r.get("group", "")).startswith("G")
            else str(r.get("group"))
        ) == group
    )

    print(f"\n{group}:")
    print("  " + "  ".join(
        f"{lang}={count}"
        for lang, count in sorted(lang_counts.items())
    ))


# ------------------------------------------------------------
# EXPECTED OUTPUT IF 5 EXAMPLES / META
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPECTED GENERATED EXAMPLES")
print("=" * 70)

print(f"5 per meta : {len(rows) * 5:,}")
print(f"6 per meta : {len(rows) * 6:,}")
print(f"10 per meta: {len(rows) * 10:,}")