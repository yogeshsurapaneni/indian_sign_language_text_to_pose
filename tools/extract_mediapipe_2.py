import os, json, hashlib, argparse
import cv2, numpy as np
from tqdm import tqdm
import mediapipe as mp

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create Holistic once (same as your script)
mp_h = mp.solutions.holistic.Holistic(static_image_mode=False)

def file_hash(path, nbytes=2_000_000):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:10]

def _landmarks_to_xy(res, include_face=False):
    """Collect (x,y) normalized landmarks from available parts in current result."""
    pts = []

    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            pts.append((lm.x, lm.y))

    if res.left_hand_landmarks:
        for lm in res.left_hand_landmarks.landmark:
            pts.append((lm.x, lm.y))

    if res.right_hand_landmarks:
        for lm in res.right_hand_landmarks.landmark:
            pts.append((lm.x, lm.y))

    if include_face and res.face_landmarks:
        for lm in res.face_landmarks.landmark:
            pts.append((lm.x, lm.y))

    return pts

def _bbox_from_points_norm(pts, min_size=0.05):
    """
    pts: list of (x,y) in normalized [0..1] (may include slightly outside).
    returns (x1,y1,x2,y2) normalized, or None.
    """
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = float(min(xs)); x2 = float(max(xs))
    y1 = float(min(ys)); y2 = float(max(ys))

    # guard against tiny / degenerate bbox
    if (x2 - x1) < min_size:
        cx = (x1 + x2) / 2.0
        x1 = cx - min_size / 2.0
        x2 = cx + min_size / 2.0
    if (y2 - y1) < min_size:
        cy = (y1 + y2) / 2.0
        y1 = cy - min_size / 2.0
        y2 = cy + min_size / 2.0

    return (x1, y1, x2, y2)

def _expand_clamp_bbox_norm(b, pad_x=0.25, pad_y=0.35):
    """Expand bbox (normalized) by fractions of its size, then clamp to [0,1]."""
    x1,y1,x2,y2 = b
    w = x2 - x1
    h = y2 - y1

    x1 -= pad_x * w
    x2 += pad_x * w
    y1 -= pad_y * h
    y2 += pad_y * h

    # clamp
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    # ensure non-degenerate
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-3)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-3)

    return (x1,y1,x2,y2)

def _ema_bbox(prev, cur, alpha=0.2):
    """EMA smoothing on bbox coords."""
    if prev is None:
        return cur
    return tuple((1 - alpha) * p + alpha * c for p, c in zip(prev, cur))

def _norm_bbox_to_px(b, W, H):
    x1,y1,x2,y2 = b
    px1 = int(round(x1 * W))
    py1 = int(round(y1 * H))
    px2 = int(round(x2 * W))
    py2 = int(round(y2 * H))

    px1 = max(0, min(W-1, px1))
    py1 = max(0, min(H-1, py1))
    px2 = max(1, min(W,   px2))  # allow W
    py2 = max(1, min(H,   py2))  # allow H

    if px2 <= px1:
        px2 = min(W, px1 + 1)
    if py2 <= py1:
        py2 = min(H, py1 + 1)

    return px1, py1, px2, py2

def extract(video_path, center_subject=True, pad_x=0.25, pad_y=0.35, smooth_alpha=0.2, include_face_in_box=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 100.0

    frames = []
    rois = []  # per-frame ROI transform (only filled if center_subject)

    missL = 0
    missR = 0
    missP = 0

    # Carry-forward buffers (NOTE: these are in ROI-normalized coords when centering is on)
    lastP = [0.0] * (33 * 3)
    lastL = [0.0] * (21 * 3)
    lastR = [0.0] * (21 * 3)
    lastF = [0.0] * (468 * 3)

    last_bbox = None  # normalized bbox on original frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]

        if not center_subject:
            # your original behavior
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_h.process(rgb)

            if res.pose_landmarks:
                P = []
                for lm in res.pose_landmarks.landmark:
                    P += [lm.x, lm.y, lm.z]
                lastP = P
            else:
                missP += 1
                P = lastP

            if res.left_hand_landmarks:
                L = []
                for lm in res.left_hand_landmarks.landmark:
                    L += [lm.x, lm.y, lm.z]
                lastL = L
            else:
                missL += 1
                L = lastL

            if res.right_hand_landmarks:
                R = []
                for lm in res.right_hand_landmarks.landmark:
                    R += [lm.x, lm.y, lm.z]
                lastR = R
            else:
                missR += 1
                R = lastR

            if res.face_landmarks:
                F = []
                for lm in res.face_landmarks.landmark:
                    F += [lm.x, lm.y, lm.z]
                lastF = F
            else:
                F = lastF

            frames.append(P + L + R + F)
            continue

        # --------------------------
        # Centering mode (2-pass):
        # pass1: locate subject bbox
        # pass2: crop ROI, extract landmarks in ROI space
        # --------------------------

        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_full = mp_h.process(rgb_full)

        pts = _landmarks_to_xy(res_full, include_face=include_face_in_box)
        bbox = _bbox_from_points_norm(pts)

        if bbox is None:
            # if detection failed: reuse last bbox if we have it
            bbox = last_bbox

        if bbox is None:
            # still nothing: fallback to center crop
            bbox = (0.25, 0.10, 0.75, 0.95)

        bbox = _expand_clamp_bbox_norm(bbox, pad_x=pad_x, pad_y=pad_y)
        bbox = _ema_bbox(last_bbox, bbox, alpha=smooth_alpha)
        last_bbox = bbox

        x1, y1, x2, y2 = _norm_bbox_to_px(bbox, W, H)
        roi_bgr = frame[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            # fallback to full frame if crop goes wrong
            roi_bgr = frame
            x1, y1, x2, y2 = 0, 0, W, H

        rgb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb_roi)

        # Save transform for this frame
        rois.append({
            "x1": int(x1), "y1": int(y1),
            "w": int(x2 - x1), "h": int(y2 - y1),
            "orig_w": int(W), "orig_h": int(H),
        })

        # Extract landmarks in ROI-normalized coords (mp returns normalized to input image)
        if res.pose_landmarks:
            P = []
            for lm in res.pose_landmarks.landmark:
                P += [lm.x, lm.y, lm.z]
            lastP = P
        else:
            missP += 1
            P = lastP

        if res.left_hand_landmarks:
            L = []
            for lm in res.left_hand_landmarks.landmark:
                L += [lm.x, lm.y, lm.z]
            lastL = L
        else:
            missL += 1
            L = lastL

        if res.right_hand_landmarks:
            R = []
            for lm in res.right_hand_landmarks.landmark:
                R += [lm.x, lm.y, lm.z]
            lastR = R
        else:
            missR += 1
            R = lastR

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
    meta = {
        "fps": float(fps),
        "frames": int(arr.shape[0]),
        "miss_pose": int(missP),
        "miss_left": int(missL),
        "miss_right": int(missR),
        "center_subject": bool(center_subject),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "smooth_alpha": float(smooth_alpha),
    }

    roi_meta = None
    if center_subject:
        # A small summary + per-frame ROI list
        if rois:
            ws = [r["w"] for r in rois]
            hs = [r["h"] for r in rois]
            meta["roi_w_mean"] = float(np.mean(ws))
            meta["roi_h_mean"] = float(np.mean(hs))
            meta["roi_w_min"] = int(np.min(ws))
            meta["roi_h_min"] = int(np.min(hs))
            meta["roi_w_max"] = int(np.max(ws))
            meta["roi_h_max"] = int(np.max(hs))
        roi_meta = rois

    return arr, meta, roi_meta

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic pose+hands from sign clips.")
    parser.add_argument("--input", default=os.path.join(PROJECT, "isl"), help="Input root (e.g., isl/)")
    parser.add_argument("--output", default=os.path.join(PROJECT, "dataset"), help="Output root (e.g., dataset/)")
    parser.add_argument("--language", default="ind", help="Language code (e.g., ind/asl/bsl)")
    parser.add_argument("--letters", default="", help="Comma-separated letter folders to process (e.g., A,B)")
    parser.add_argument("--glosses", default="", help="Comma-separated gloss names to process (e.g., Abacus,Abdicate)")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip if pose.npy already exists")

    # NEW: centering controls
    parser.add_argument("--center", action="store_true", help="Center crop on subject before pose extraction")
    parser.add_argument("--pad-x", type=float, default=0.25, help="Horizontal padding fraction around subject bbox")
    parser.add_argument("--pad-y", type=float, default=0.35, help="Vertical padding fraction around subject bbox")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="EMA smoothing alpha for bbox (0..1)")
    parser.add_argument("--face-in-box", action="store_true", help="Include face landmarks when computing bbox")
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

            gloss = os.path.splitext(fn)[0]
            if gloss_filter and gloss not in gloss_filter:
                continue
            vpath = os.path.join(lpath, fn)

            cid = file_hash(vpath)
            outdir = os.path.join(out_root, language, letter.upper(), gloss, cid)

            if args.no_overwrite and os.path.exists(os.path.join(outdir, "pose.npy")):
                continue

            pose, meta, roi_meta = extract(
                vpath,
                center_subject=args.center,
                pad_x=args.pad_x,
                pad_y=args.pad_y,
                smooth_alpha=args.smooth_alpha,
                include_face_in_box=args.face_in_box,
            )
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

            # NEW: save ROI transforms when centering is enabled
            if roi_meta is not None:
                with open(os.path.join(outdir, "roi.json"), "w", encoding="utf-8") as f:
                    json.dump(roi_meta, f, indent=2)

    print("✅ Extraction complete. Data written to:", out_root)

if __name__ == "__main__":
    main()
