import os, json, argparse

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(PROJECT, "dataset")
OUT = os.path.join(PROJECT, "configs", "glossary.json")

def main():
    parser = argparse.ArgumentParser(description="Build glossary.json from processed dataset glosses.")
    parser.add_argument("--language", default="ind", help="Language code (e.g., ind/asl/bsl)")
    parser.add_argument("--output", default=OUT, help="Output glossary path")
    args = parser.parse_args()

    language = args.language.lower()
    lroot = os.path.join(DATASET, language)
    if not os.path.isdir(lroot):
        raise SystemExit(f"Dataset language folder not found: {lroot}")

    glossary = {}

    for letter in sorted(os.listdir(lroot)):
        lpath = os.path.join(lroot, letter)
        if not os.path.isdir(lpath):
            continue

        for gloss in sorted(os.listdir(lpath)):
            gpath = os.path.join(lpath, gloss)
            if not os.path.isdir(gpath):
                continue

            key = gloss.lower()
            if key not in glossary:
                glossary[key] = [gloss]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(glossary)} entries to {args.output}")

if __name__ == "__main__":
    main()
