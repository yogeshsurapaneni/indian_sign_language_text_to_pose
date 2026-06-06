import os, json, hashlib, argparse
import cv2, numpy as np
from tqdm import tqdm
import mediapipe as mp

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mp_h = mp.solutions.holistic.Holistic(static_image_mode=False)

def file_hash(path, nbytes=2_000_000):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:10]

def extract(video_path):
    cap = cv2.VideoCapture(video_path)
    print(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 100.0

    frames = []
    missL = 0
    missR = 0
    missP = 0

    lastP = [0.0] * (33 * 3)
    lastL = [0.0] * (21 * 3)
    lastR = [0.0] * (21 * 3)
    lastF = [0.0] * (468 * 3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb)

        # pose (required) - carry-forward if missing
        if res.pose_landmarks:
            P = []
            for lm in res.pose_landmarks.landmark:
                P += [lm.x, lm.y, lm.z]
            lastP = P
        else:
            missP += 1
            P = lastP

        # left hand (optional)
        if res.left_hand_landmarks:
            L = []
            for lm in res.left_hand_landmarks.landmark:
                L += [lm.x, lm.y, lm.z]
            lastL = L
        else:
            missL += 1
            L = lastL

        # right hand (optional)
        if res.right_hand_landmarks:
            R = []
            for lm in res.right_hand_landmarks.landmark:
                R += [lm.x, lm.y, lm.z]
            lastR = R
        else:
            missR += 1
            R = lastR

        # face (optional)
        if res.face_landmarks:
            F = []
            for lm in res.face_landmarks.landmark:
                F += [lm.x, lm.y, lm.z]
            lastF = F
        else:
            F = lastF

        frames.append(P + L + R + F)

    cap.release()
    arr = np.array(frames, dtype=np.float32)
    meta = {"fps": fps, "frames": int(arr.shape[0]), "miss_pose": missP, "miss_left": missL, "miss_right": missR}
    return arr, meta

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic pose+hands from sign clips.")
    parser.add_argument("--input", default=os.path.join(PROJECT, "isl"), help="Input root (e.g., isl/)")
    parser.add_argument("--output", default=os.path.join(PROJECT, "dataset"), help="Output root (e.g., dataset/)")
    parser.add_argument("--language", default="ind", help="Language code (e.g., ind/asl/bsl)")
    parser.add_argument("--letters", default="", help="Comma-separated letter folders to process (e.g., A,B)")
    parser.add_argument("--glosses", default="", help="Comma-separated gloss names to process (e.g., Abacus,Abdicate)")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip if pose.npy already exists")
    args = parser.parse_args()

    in_root = args.input
    out_root = args.output
    language = args.language.lower()

    os.makedirs(out_root, exist_ok=True)

    letter_filter = set()
    if args.letters.strip():
        letter_filter = {x.strip().upper() for x in args.letters.split(",") if x.strip()}

    gloss_filter = set()
    if args.glosses.strip():
        gloss_filter = {x.strip() for x in args.glosses.split(",") if x.strip()}

    # Input folder: isl/A/Absent.mp4
    for letter in sorted(os.listdir(in_root)):
        if letter_filter and letter.upper() not in letter_filter:
            continue
        lpath = os.path.join(in_root, letter)
        if not os.path.isdir(lpath):
            continue

        print(f"[EXTRACT] {letter}")
        for fn in tqdm(sorted(os.listdir(lpath))):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in [".mp4", ".m4v", ".mpg"]:
                continue

            gloss = os.path.splitext(fn)[0]          # Absent
            if gloss_filter and gloss not in gloss_filter:
                continue
            vpath = os.path.join(lpath, fn)

            # clip id based on file hash so re-runs don't duplicate
            cid = file_hash(vpath)
            outdir = os.path.join(out_root, language, letter.upper(), gloss, cid)

            if args.no_overwrite and os.path.exists(os.path.join(outdir, "pose.npy")):
                continue  # already processed

            print(vpath)
            pose, meta = extract(vpath)
            if pose.shape[0] < 1:
                continue

            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, "pose.npy"), pose)

            with open(os.path.join(outdir, "gloss.txt"), "w", encoding="utf-8") as f:
                f.write(gloss)

            rel_src = os.path.relpath(vpath, PROJECT)
            meta.update({
                "language": language,
                "letter": letter.upper(),
                "gloss": gloss,
                "clip_id": cid,
                "source": rel_src
            })
            with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    print("✅ Extraction complete. Data written to:", out_root)

if __name__ == "__main__":
    main()
