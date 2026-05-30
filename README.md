# ML Google reCAPTCHA v2 Solver

An end-to-end machine learning pipeline that automatically solves Google reCAPTCHA v2 image challenges. The system collects its own training data, trains a tile classifier, and deploys a local server that a Chrome extension calls to solve live CAPTCHAs.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline](#pipeline)
   - [1. Data Collection](#1-data-collection)
   - [2. Data Cleaning](#2-data-cleaning)
   - [3. Training](#3-training)
   - [4. Export](#4-export)
   - [5. Deployment](#5-deployment)
4. [Design Decisions](#design-decisions)
5. [Quick Start](#quick-start)

---

## Overview

reCAPTCHA v2 image challenges ask users to click all tiles in a grid that contain a specific object — bicycles, traffic lights, crosswalks, etc. The challenge comes in two formats:

- **3×3 (9 tiles):** A single large image segmented into 9 pieces. 
- **4×4 (16 tiles):** A single large image segmented into 16 pieces.

This project solves both formats using different strategies (classifier for 3×3, YOLO object detection for 4×4), with a browser automation layer that handles clicking, waiting for tile transitions, and injecting the resulting token back into the original page.

---

## Architecture

```
Chrome Extension (content.js + background.js)
        │
        │  POST /solve  {pageurl}
        ▼
Local Flask Server (solver/server.py)
        │
        ├── Persistent Playwright/Chromium browser (pre-launched at startup)
        │       └── Navigates to pageurl, clicks CAPTCHA checkbox
        │           Detects challenge type (3×3 or 4×4)
        │           Waits for tiles to fully load + fade in
        │           Runs inference → clicks tiles → verifies
        │           Returns g-recaptcha-response token
        │
        ├── MobileNetV3-Small ONNX (3×3 tile classification)
        └── YOLOWorld (4×4 full-image object detection)
```

The extension is a thin relay: it detects when a reCAPTCHA appears on any page, sends the page URL to the local server, and injects the returned token into the `g-recaptcha-response` textarea.

---

## Pipeline

### 1. Data Collection

**`data-collection/scrape.py`**

The scraper navigates to the reCAPTCHA demo page, triggers the challenge, reads the instruction text to identify the category (e.g. "Select all images with **bicycles**"), and screenshots each individual tile. Tiles are saved to `data-collection/images/<category>/`.

Key decisions:
- **Playwright + stealth patches**: patches `navigator.webdriver`, `navigator.plugins`, `window.chrome`, and the permissions API so reCAPTCHA's bot detection doesn't immediately reject the browser.
- **Rotating user agents, viewports, locales, and timezones**: each worker gets a randomized fingerprint to avoid a single detectable profile.
- **Proxy support**: accepts a proxy list (`proxies.txt`) in `host:port:user:pass` format to distribute requests across IPs. Without proxies, Google rate-limits aggressive scraping.
- **Parallel workers**: runs N browsers concurrently via asyncio, each scraping independently, with a shared atomic counter to coordinate progress.

```bash
# 20 parallel browsers with proxies
python scrape.py --workers 20 --proxy-file proxies.txt
```

---

### 2. Data Cleaning

**`data-collection/clean.py`**

Raw scraped tiles are noisy and reCAPTCHA occasionally shows ambiguous or mislabeled images. Our three-model cleaning pipeline filters them:

| Model | Categories | Rationale |
|---|---|---|
| `yolo11n.pt` (COCO 80) | bicycles, boats, buses, cars, hydrants, lights, meters, motorcycles | Fast, accurate detection for common objects |
| `yolov8n-oiv7.pt` (Open Images V7, 601 classes) | stairs | COCO doesn't include stairs whereas OIV7 does |
| CLIP `ViT-B/32` | bridges, chimneys, crosswalks, hills | Scene-level categories with no reliable detection box. CLIP's image-text similarity scores scene context better than a bounding box model |

Each tile is kept only if the appropriate model detects the target with sufficient confidence. Rejected tiles move to `data-collection/rejected/<category>/` and can be restored with `--restore`.

```bash
python clean.py \
  --yolo bicycles boats buses cars hydrants lights meters motorcycles \
  --oiv7 stairs \
  --clip bridges chimneys crosswalks hills \
  --clip-thresh 0.24
```

---

### 3. Training

**`training/train.py`**

Trains a **MobileNetV3-Small** image classifier (pretrained on ImageNet, fine-tuned on our tile dataset) across 15 categories. MobileNetV3-Small was chosen for its balance of accuracy and size — the exported ONNX model is ~5.9 MB, small enough to load quickly and run efficiently on CPU.

**Class balancing via `WeightedRandomSampler`**

Some categories are inherently rarer in reCAPTCHA challenges (hills, chimneys, tractors) while others appear constantly (cars, traffic lights). Without correction, the model overfits to majority classes. A `WeightedRandomSampler` oversamples minority classes so every class appears roughly equally per epoch where each epoch draws `max_class_count × num_classes` samples with replacement.

**Augmentation pipeline**

Because tiles are small (~100–150px) and often visually similar, aggressive augmentation is essential:

- `RandomResizedCrop(224, scale=(0.65, 1.0))`: spatial variety without always centering the object
- `RandomHorizontalFlip` + `RandomVerticalFlip(p=0.15)`: orientational diversity (aerial shots can face any direction)
- `RandomRotation(20°)`: handles tilted captures
- `RandomPerspective(distortion_scale=0.25)`: simulates different camera angles
- `ColorJitter(brightness, contrast, saturation, hue)`: robustness to varying lighting and compression artifacts
- `GaussianBlur`: handles blurry or low-quality tiles from compressed JPEG sources
- `RandomGrayscale(p=0.08)`: helps with desaturated or nighttime images
- `RandomErasing(p=0.2)`: partial occlusion robustness (tiles often have overlapping UI elements)

**Training details:**
- Optimizer: AdamW with cosine annealing LR schedule
- Loss: CrossEntropyLoss with label smoothing (0.1) to prevent overconfidence
- Best validation accuracy reached: **~94.3%** at epoch 22 of 25

```bash
python train.py --epochs 25 --batch 64
```

---

### 4. Export

**`training/export.py`**

Exports the trained PyTorch checkpoint to ONNX format for deployment. ONNX was chosen so the model runs with `onnxruntime` on CPU without requiring PyTorch at inference time, keeping the server's startup fast and its dependency footprint small.

The exporter also writes `class_map.json` alongside the model so the server knows which output index corresponds to which category.

```bash
python export.py  # model.pt → ../extension/model.onnx
```

---

### 5. Deployment

**`solver/server.py` + `extension/`**

#### Server

A Flask server listens on `http://localhost:5000`. On startup it:
1. Loads the ONNX classifier and (optionally) the YOLOWorld model
2. Launches a persistent Chromium browser via Playwright. This is the key performance optimization: browser startup takes 3–5 seconds, so launching once at startup and reusing the same page eliminates per-request overhead
3. Optionally loads a saved Google session (`session.json`) so the browser is already authenticated. Google assigns each session a trust score between 0 and 0.9, and sessions below roughly 0.6 receive a harder challenge variant: tiles fade in and out dynamically and contain more visual noise, both of which reduce classifier accuracy. Higher-trust accounts see cleaner, more static grids. Trust builds naturally through regular use of Google services (search, YouTube, Gmail), so an aged, active account solves more reliably than a fresh or anonymous one.

Per request (`POST /solve`):
1. Navigates the persistent page to `pageurl`
2. Clicks the CAPTCHA checkbox
3. Detects the grid type from tile count (9 = 3×3, 16 = 4×4)
4. Waits for all tiles to fully load **and** finish their CSS fade-in animation (checked via computed `opacity > 0.95` on the `<td>` elements)
5. Runs inference, clicks tiles with human-paced random delays
6. Polls for tile replacements for up to 3 seconds after each click. Any changed tile src URL triggers re-classification before verifying
7. Returns the `g-recaptcha-response` token

**3×3 strategy: MobileNetV3 classifier**

Each tile is cropped from a single bframe screenshot and classified independently. The confidence threshold is 0.8 (configurable via `--conf`). A sanity check rejects rounds where >60% of tiles exceed the threshold, which usually indicates the model is biased toward that category; the challenge is reloaded instead.

**4×4 strategy: YOLOWorld detection**

Because a 4×4 challenge is one image split into 16 segments, classifying each tiny tile independently loses spatial context. Instead, all 16 tiles are stitched back into one composite image, and `YOLOWorld` is run with the target category as a text prompt (e.g. `"fire hydrant"`, `"traffic light"`). Each detected bounding box is intersected with the 16 tile cells; a tile's score is `detection_confidence × overlap_fraction`. The confidence threshold for 4×4 is 0.1 (much lower than 3×3 because YOLO scores represent overlap, not classification probability).

**Anti-detection measures**
- Stealth JS patches on every new page context (hides `navigator.webdriver`, adds fake `navigator.plugins`, spoofs `window.chrome`)
- Random human-paced delays between tile clicks (0.4–0.9 s) and a thinking pause before clicking (0.8–1.4 s)
- Resource blocking for non-reCAPTCHA assets (speeds up page load without touching reCAPTCHA's own JS/images)
- Proxy support (`--proxy host:port:user:pass`)

#### Chrome Extension

The extension is intentionally thin. It does the minimum needed to bridge the browser's same-origin restrictions:

- **`content.js`**: detects the reCAPTCHA anchor iframe appearing in the DOM, sends the page URL to the background service worker, and injects the returned token into `g-recaptcha-response` + fires any registered widget callbacks
- **`background.js`**: relays the HTTP request to `localhost:5000` (content scripts can't fetch `http://` from `https://` pages due to mixed-content rules; service workers bypass this)
- No bundlers, no build step — three plain JS files and a manifest

---

## Design Decisions

**Why a local server instead of running inference inside the extension?**

The original design ran the ONNX model directly in the browser using onnxruntime-web (WASM). This approach was abandoned because:

- Cold-starting the WASM runtime on each page visit introduced 2–4 seconds of visible latency.
- A local Python server provides access to the full ONNX Runtime stack (CPU and GPU execution), Playwright for browser automation, and the broader YOLO ecosystem—capabilities that are not available within a browser extension's sandbox.

**Why MobileNetV3-Small for 3×3?**

reCAPTCHA tiles are small images of common objects against varied backgrounds which is exactly the problem ImageNet-pretrained models handle well. MobileNetV3-Small reaches ~94% validation accuracy at 5.9 MB, runs in <5 ms per tile on CPU, and requires no GPU at inference time. A larger model would not meaningfully improve accuracy because the bottleneck is image count and label noise in reCAPTCHA's challenges, not model capacity.

**Why YOLOWorld for 4×4?**

A 4×4 challenge is one image segmented into 16 pieces, so the target object often spans multiple tiles. Classifying each 100px tile in isolation loses this spatial context and creates hard boundary effects. YOLOWorld detects bounding boxes on the full stitched image and maps them back to tile indices via intersection area. The World model variant accepts arbitrary text prompts, so the same model weights cover all 15 categories without any category-specific training.

**Why a persistent browser instead of one per request?**

Launching a Chromium instance takes 3–5 seconds. For a service that may be called many times, spawning a new browser per request would dominate the total solve time. A single pre-launched page is reset between requests (`unroute` + navigate to `about:blank`), cutting that overhead to near zero. An asyncio lock serializes requests so two solves never share the same page.

---

## Quick Start

### Prerequisites

```bash
pip install playwright onnxruntime flask flask-cors pillow numpy ultralytics
python -m playwright install chromium
```

### 1. Collect data

```bash
cd data-collection
python scrape.py --workers 10 --proxy-file proxies.txt
```

### 2. Clean data

```bash
python clean.py \
  --yolo bicycles boats buses cars hydrants lights meters motorcycles \
  --oiv7 stairs \
  --clip bridges chimneys crosswalks hills \
  --clip-thresh 0.24
```

### 3. Train

```bash
cd ../training
python train.py --epochs 25 --batch 64
```

### 4. Export

```bash
python export.py
```

### 5. (Optional) Save a Google session

```bash
cd ../solver
python server.py --save-session
# Log in to Google in the browser that opens, then press Enter
```

### 6. Start the server

```bash
python server.py
# With proxy: python server.py --proxy host:port:user:pass
```

### 7. Load the extension

Open Chrome → `chrome://extensions` → Enable Developer Mode → Load Unpacked → select the `extension/` folder.

The extension automatically solves supported reCAPTCHA v2 image challenges, however, performance will vary according to the model's training data (the more images collected, the better).