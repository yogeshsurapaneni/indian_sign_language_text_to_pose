import os, re, json
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(PROJECT, "dataset", "index.json")
GLOSSARY = os.path.join(PROJECT, "configs", "glossary.json")
OUT_JSON = os.path.join(PROJECT, "web_viewer", "pose.json")

DIGIT = {
    "0":"ZERO","1":"ONE","2":"TWO","3":"THREE","4":"FOUR",
    "5":"FIVE","6":"SIX","7":"SEVEN","8":"EIGHT","9":"NINE"
}

def tokenize(text: str):
    # keep numbers as tokens
    text = text.lower()
    return re.findall(r"[a-z]+|\d+", text)

def text_to_gloss(tokens, glossary):
    out = []
    for tok in tokens:
        if tok.isdigit():
            # 45464 -> FOUR FIVE FOUR SIX FOUR
            out.extend([DIGIT[d] for d in tok])
            continue
        if tok in glossary:
            out.extend(glossary[tok])
        else:
            # unknown word: skip (for now)
            pass
    return out

def pick_clip(gloss, index, language):
    # pick first available clip for gloss (case-insensitive)
    key = gloss.lower()
    if language not in index:
        return None
    if key not in index[language]["index"]:
        return None
    return index[language]["index"][key][0]["pose"]

def main(sentence: str, language: str = "ind"):
    language = language.lower()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    with open(INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)["index"]

    with open(GLOSSARY, "r", encoding="utf-8") as f:
        glossary = json.load(f)

    toks = tokenize(sentence)
    glosses = text_to_gloss(toks, glossary)

    poses = []
    used = []
    missing = []

    for g in glosses:
        p = pick_clip(g, idx, language)
        if not p:
            missing.append(g)
            continue
        arr = np.load(os.path.join(PROJECT, p))
        poses.append(arr)
        used.append(g)

    if not poses:
        raise SystemExit("No matching gloss clips found. Add more signs or extend glossary.json.")

    stitched = np.concatenate(poses, axis=0)  # (T,225)

    # export as browser-friendly json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"frames": stitched.tolist(), "gloss_sequence": used, "missing": missing, "dims": stitched.shape[1]}, f)

    print("✅ Wrote:", OUT_JSON)
    print("Used:", " ".join(used))
    if missing:
        print("Missing:", " ".join(missing))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stitch pose clips into a sentence sequence.")
    parser.add_argument("--sentence", default="Train number 45464 arriving on platform 4")
    parser.add_argument("--language", default="ind")
    args = parser.parse_args()
    main(args.sentence, language=args.language)
