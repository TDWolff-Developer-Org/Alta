import sys
import os
import csv

try:
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset
except ImportError:
    print("ERROR: Missing dependencies.")
    print("Run: pip install datasets pandas tqdm")
    sys.exit(1)

DATA_PATH = "data.csv"

# ── Limits ────────────────────────────────────────────────────────────────────
# How many examples to pull from each source.
# Lower these if you want a faster run.

CODESEARCHNET_PER_LANG = 5_000   # × 6 languages = up to 30k
CODEALPACA_LIMIT       = 20_000  # instruction/code pairs

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_existing_prompts(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["prompt"].astype(str).str.strip())


def append_rows(path, rows):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writerow(["prompt", "completion", "label"])
        for prompt, completion in rows:
            writer.writerow([prompt, completion, 1])


def is_usable(text, min_len=10, max_len=1500):
    return text and min_len <= len(text.strip()) <= max_len

# ── Sources ───────────────────────────────────────────────────────────────────

def collect_codesearchnet(existing, per_lang=5_000):
    languages = ["python", "javascript", "java", "go", "php", "ruby"]
    rows = []

    for lang in languages:
        print(f"\n  [{lang}] loading...")
        try:
            ds = load_dataset(
                "code_search_net", lang,
                split="train",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [{lang}] failed: {e}")
            continue

        count = 0
        for item in tqdm(ds, desc=f"  [{lang}]", unit="ex"):
            if count >= per_lang:
                break

            doc  = (item.get("func_documentation_string") or "").strip()
            code = (item.get("func_code_string") or "").strip()

            if not is_usable(doc, 15, 400) or not is_usable(code, 30, 1200):
                continue

            # Turn docstring into a natural prompt
            prompt = f"Write a {lang} function that: {doc}"
            if prompt in existing:
                continue

            rows.append((prompt, code))
            existing.add(prompt)
            count += 1

        print(f"  [{lang}] collected {count} examples")

    return rows


def collect_codealpaca(existing, limit=20_000):
    print("\n  [CodeAlpaca] loading...")
    try:
        ds = load_dataset(
            "sahil2801/CodeAlpaca-20k",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [CodeAlpaca] failed: {e}")
        return []

    rows = []
    for item in tqdm(ds, desc="  [CodeAlpaca]", unit="ex"):
        if len(rows) >= limit:
            break

        instruction = (item.get("instruction") or "").strip()
        extra       = (item.get("input") or "").strip()
        output      = (item.get("output") or "").strip()

        if not is_usable(instruction, 10, 400) or not is_usable(output, 10, 1500):
            continue

        prompt = f"{instruction} {extra}".strip() if extra else instruction
        if prompt in existing:
            continue

        rows.append((prompt, output))
        existing.add(prompt)

    print(f"  [CodeAlpaca] collected {len(rows)} examples")
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"── Loading existing data from {DATA_PATH} ────────────────────────")
    existing = load_existing_prompts(DATA_PATH)
    print(f"  {len(existing)} existing examples\n")

    all_rows = []

    print("── CodeSearchNet ─────────────────────────────────────────────────")
    all_rows.extend(collect_codesearchnet(existing, CODESEARCHNET_PER_LANG))

    print("\n── CodeAlpaca ────────────────────────────────────────────────────")
    all_rows.extend(collect_codealpaca(existing, CODEALPACA_LIMIT))

    print(f"\n── Saving ────────────────────────────────────────────────────────")
    if not all_rows:
        print("  Nothing new to add.")
        return

    print(f"  Appending {len(all_rows):,} new rows to {DATA_PATH}...")
    append_rows(DATA_PATH, all_rows)

    total = len(existing) + len(all_rows)
    print(f"  Done. data.csv now has ~{total:,} examples.")
    print(f"\n  Breakdown:")
    print(f"    Previous : {len(existing):>7,}")
    print(f"    New      : {len(all_rows):>7,}")
    print(f"    Total    : {total:>7,}")


if __name__ == "__main__":
    main()
