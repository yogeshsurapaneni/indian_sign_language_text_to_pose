import os, json

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(PROJECT, "dataset")
MANIFEST = os.path.join(DATASET, "manifest.jsonl")
INDEX = os.path.join(DATASET, "index.json")

def main():
    os.makedirs(DATASET, exist_ok=True)

    rows = []
    lang_index = {}

    for language in sorted(os.listdir(DATASET)):
        lroot = os.path.join(DATASET, language)
        if not os.path.isdir(lroot):
            continue

        gloss_index = {}

        for letter in sorted(os.listdir(lroot)):
            lpath = os.path.join(lroot, letter)
            if not os.path.isdir(lpath):
                continue

            for gloss in sorted(os.listdir(lpath)):
                gpath = os.path.join(lpath, gloss)
                if not os.path.isdir(gpath):
                    continue

                for clip_id in sorted(os.listdir(gpath)):
                    cpath = os.path.join(gpath, clip_id)
                    pose_path = os.path.join(cpath, "pose.npy")
                    meta_path = os.path.join(cpath, "meta.json")
                    if not os.path.exists(pose_path):
                        continue

                    meta = {}
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)

                    row = {
                        "language": language,
                        "letter": letter,
                        "gloss": gloss,
                        "clip_id": clip_id,
                        "pose": os.path.relpath(pose_path, PROJECT),
                        "meta": os.path.relpath(meta_path, PROJECT) if os.path.exists(meta_path) else None
                    }
                    rows.append(row)
                    gloss_index.setdefault(gloss.lower(), []).append(row)

        lang_index[language] = {
            "glosses": sorted(gloss_index.keys()),
            "index": gloss_index
        }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump({"languages": sorted(lang_index.keys()), "index": lang_index}, f, indent=2)

    print(f"✅ Wrote {len(rows)} clips to {MANIFEST}")
    print(f"✅ Gloss index at {INDEX}")

if __name__ == "__main__":
    main()
