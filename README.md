<div align="center">
  <h1>👋 IndianSigner: Text-to-Pose for ISL</h1>
  <p><strong>A modular pipeline for extracting, stitching, and visualizing Indian Sign Language (ISL) data.</strong></p>
</div>

---

## 📖 Overview

**IndianSigner** is a complete, end-to-end toolchain designed to process Indian Sign Language (ISL) video datasets into 3D skeletal animations. 

It provides an efficient pipeline to:
1. **Extract** 3D landmarks (Pose, Hands, Face) from sign language video clips using Google MediaPipe Holistic.
2. **Translate** English text into ISL-compatible sequences using gloss mapping.
3. **Stitch** multiple sign poses into a continuous, fluid animation array.
4. **Visualize** the final synthesized sequences in a lightweight 3D Web Viewer.

## ✨ Features

- **Robust Pose Extraction**: Extracts comprehensive landmark data (up to 543 points per frame) using `mediapipe`. Includes experimental subject tracking and centering with EMA smoothing for stable extraction.
- **Smart Text-to-Gloss Mapping**: Tokenizes English sentences, applies digit expansion, and maps words to ISL video clips using customizable `glossary.json`.
- **Dynamic Stitching**: Concatenates multiple `.npy` pose sequences dynamically, generating a combined JSON sequence optimized for web playback.
- **3D Web Viewer**: Includes an interactive browser-based viewer built with HTML/JS/CSS to visualize the stitched `pose.json` arrays in real-time.
- **Dataset Management**: Tools to build manifests (`build_manifest.py`), generate glossaries (`build_glossary.py`), and ingest from cloud storage (`ingest_gdrive.py`).

## 🗂️ Project Structure

```text
indian_sign_language_text_to_pose/
├── configs/            # Configuration files (e.g., glossary.json, index mappings)
├── dataset/            # Processed output dataset (extracted .npy poses, manifests)
├── isl/                # Raw ISL video clips organized by letters/categories
├── tools/              # Python scripts for data extraction, stitching, and management
│   ├── build_glossary.py
│   ├── extract_mediapipe_2.py
│   ├── stitch_sentence.py
│   └── ...
└── web_viewer/         # HTML/JS/CSS files for 3D browser visualization
```

## 🚀 Getting Started

### Prerequisites

You need **Python 3.8+** installed. Install the necessary dependencies:

```bash
pip install numpy opencv-python mediapipe tqdm
```

### 1. Data Extraction

Convert raw ISL video clips into normalized `.npy` pose sequences using the MediaPipe extractor. Place your raw `.mp4` videos in the `isl/` directory.

```bash
# Basic extraction
python tools/extract_mediapipe_2.py --input isl --output dataset --language ind

# Advanced extraction with subject centering and EMA smoothing
python tools/extract_mediapipe_2.py --center --smooth-alpha 0.2
```

### 2. Sentence Stitching

Translate a given text sentence into a continuous ISL pose sequence. The script maps tokens to glosses and concatenates the corresponding `.npy` arrays, exporting a final `pose.json`.

```bash
python tools/stitch_sentence.py --sentence "Train number 45464 arriving on platform 4" --language ind
```
*Note: Ensure your `dataset/index.json` and `configs/glossary.json` are populated correctly for mapping.*

### 3. Web Visualization

To view the generated `pose.json` animation:
1. Open a local development server in the `web_viewer/` directory:
   ```bash
   cd web_viewer
   python -m http.server 8000
   ```
2. Navigate to `http://localhost:8000` in your web browser.

## 🛠️ Contribution & Development

Contributions, issues, and feature requests are welcome! 
Feel free to check the [issues page](../../issues) if you want to contribute.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
