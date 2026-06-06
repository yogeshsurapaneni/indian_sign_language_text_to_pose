import os, json, re, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT, "raw")
SOURCES = os.path.join(PROJECT, "configs", "sources.json")

def folder_id_from_url(url: str) -> str:
    # supports either full URL or raw ID
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else url.strip()

def run(cmd):
    print(">", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise SystemExit(p.returncode)
    return p.stdout

def main():
    os.makedirs(RAW, exist_ok=True)
    with open(SOURCES, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for folder in cfg.get("gdrive_folders", []):
        fid = folder_id_from_url(folder)
        # gdown --folder downloads recursively.
        # --remaining-ok allows continuing partial downloads.
        # --no-cookies helps with some public link cases.
        cmd = [
            sys.executable, "-m", "gdown",
            "--folder", fid,
            "--output", RAW,
            "--remaining-ok",
            "--no-cookies"
        ]
        run(cmd)

    print("✅ Ingest complete. Raw videos are in:", RAW)
    print("Expected structure: raw/A/*.mp4, raw/B/*.mp4, ...")

if __name__ == "__main__":
    main()
