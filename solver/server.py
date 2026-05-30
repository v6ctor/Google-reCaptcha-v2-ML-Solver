"""
Local reCAPTCHA solver server.

Usage:
    python server.py
    python server.py --model ../training/model.onnx --port 5000

The Chrome extension POSTs {sitekey, pageurl} to /solve.
The server uses a persistent Chromium browser launched at startup,
solves the tile challenge using the ONNX model, and returns the
g-recaptcha-response token.
"""

import argparse
import asyncio
import json
import random
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from playwright.async_api import async_playwright

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONF_THRESHOLD  = 0.4
MAX_ROUNDS      = 20    # max challenge rounds before giving up
CAPTCHA_W       = 300   # reCAPTCHA bframe width  (same for all challenges)
CAPTCHA_H       = 620   # reCAPTCHA bframe height (same for all challenges)
SKIP_CATEGORIES = {}

KEYWORD_TO_CATEGORY = {
    "bicycle": "bicycles", "bicycles": "bicycles",
    "boat": "boats", "boats": "boats",
    "bridge": "bridges", "bridges": "bridges",
    "bus": "buses", "buses": "buses",
    "car": "cars", "cars": "cars",
    "chimney": "chimneys", "chimneys": "chimneys",
    "crosswalk": "crosswalks", "crosswalks": "crosswalks",
    "hill": "hills", "hills": "hills",
    "hydrant": "hydrants", "hydrants": "hydrants",
    "fire hydrant": "hydrants",
    "traffic light": "lights", "traffic lights": "lights",
    "light": "lights", "lights": "lights",
    "meter": "meters", "meters": "meters",
    "parking meter": "meters",
    "motorcycle": "motorcycles", "motorcycles": "motorcycles",
    "stair": "stairs", "stairs": "stairs",
}

session: ort.InferenceSession = None
class_map: dict = None   # {"0": "bicycles", ...}
_headless: bool = False
_yolo = None             # Optional YOLOWorld model for 4×4 detection

# reCAPTCHA category → YOLO text prompt used with the World model
CATEGORY_TO_PROMPT = {
    "bicycles":   "bicycle",
    "boats":      "boat",
    "bridges":    "bridge",
    "buses":      "bus",
    "cars":       "car",
    "chimneys":   "chimney",
    "crosswalks": "crosswalk",
    "hills":      "hill",
    "hydrants":   "fire hydrant",
    "lights":     "traffic light",
    "meters":     "parking meter",
    "motorcycles":"motorcycle",
    "stairs":     "stairs",
}


def load_model(model_path: Path, class_map_path: Path):
    global session, class_map
    session = ort.InferenceSession(str(model_path),
                                   providers=["CPUExecutionProvider"])
    class_map = json.loads(class_map_path.read_text())
    print(f"  Model loaded: {model_path}  ({model_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  Classes: {list(class_map.values())}")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 224, 224]


def classify(img: Image.Image) -> np.ndarray:
    logits = session.run(None, {"image": preprocess(img)})[0][0]
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()   # softmax probabilities


def load_yolo(model_path: Path):
    global _yolo
    from ultralytics import YOLOWorld
    _yolo = YOLOWorld(str(model_path))
    print(f"  YOLO model loaded: {model_path}")


def detect_4x4(frame_img: Image.Image, tile_boxes: list[dict], category: str) -> list[float]:
    """
    Stitch the 4×4 tile grid into one image, run YOLO World detection,
    then compute per-tile scores as (max detection confidence × overlap ratio).
    Returns a list of 16 floats, or None if YOLO is unavailable for this category.
    """
    if _yolo is None or category not in CATEGORY_TO_PROMPT:
        return None

    prompt = CATEGORY_TO_PROMPT[category]
    sorted_boxes = sorted(tile_boxes, key=lambda b: (b["y"], b["x"]))
    tw = int(sorted_boxes[0]["width"])
    th = int(sorted_boxes[0]["height"])

    # Build composite 4×4 image from per-tile crops
    stitched = Image.new("RGB", (4 * tw, 4 * th))
    for i, b in enumerate(sorted_boxes):
        r, c = divmod(i, 4)
        crop = frame_img.crop((int(b["x"]), int(b["y"]),
                               int(b["x"]) + tw, int(b["y"]) + th))
        stitched.paste(crop, (c * tw, r * th))

    # Run YOLO with the target class as the text prompt
    _yolo.set_classes([prompt])
    results = _yolo.predict(stitched, verbose=False, conf=0.1)

    tile_scores = [0.0] * 16
    if results and results[0].boxes is not None and len(results[0].boxes):
        tile_area = tw * th
        for box in results[0].boxes:
            bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0]]
            det_conf = float(box.conf[0])
            for i in range(16):
                r, c = divmod(i, 4)
                tx1, ty1 = c * tw, r * th
                tx2, ty2 = tx1 + tw, ty1 + th
                ix1, iy1 = max(bx1, tx1), max(by1, ty1)
                ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = ((ix2 - ix1) * (iy2 - iy1)) / tile_area
                    tile_scores[i] = max(tile_scores[i], det_conf * overlap)

    return tile_scores

def extract_category(text: str) -> str | None:
    lower = text.lower()
    # Try longest phrases first
    for phrase in sorted(KEYWORD_TO_CATEGORY, key=len, reverse=True):
        if phrase in lower:
            return KEYWORD_TO_CATEGORY[phrase]
    return None


def category_to_index(category: str) -> int | None:
    for k, v in class_map.items():
        if v == category:
            return int(k)
    return None


BLOCK_RESOURCE_TYPES = {"image", "stylesheet", "font", "media", "ping", "other"}
RECAPTCHA_HOSTS     = {"www.google.com", "www.gstatic.com", "recaptcha.google.com"}


async def _block_non_recaptcha(route):
    """Abort heavy page assets; let reCAPTCHA resources through."""
    req = route.request
    if req.resource_type in BLOCK_RESOURCE_TYPES:
        from urllib.parse import urlparse
        host = urlparse(req.url).hostname or ""
        if host not in RECAPTCHA_HOSTS:
            await route.abort()
            return
    await route.continue_()

_browser_loop: asyncio.AbstractEventLoop = None
_pw        = None   # Playwright instance
_browser   = None   # Chromium browser
_page      = None   # single persistent page
_browser_lock = asyncio.Lock()   # created inside the browser loop
_proxy: dict | None = None   # {"server": "...", "username": "...", "password": "..."}
_session_path: Path | None = None  # path to saved Playwright storage-state JSON

_STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){}, app: {} };
    const orig = window.navigator.permissions.query.bind(window.navigator.permissions);
    window.navigator.permissions.query = (p) =>
        p.name === 'notifications'
            ? Promise.resolve({ state: Notification.permission })
            : orig(p);
"""


def _run_event_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _launch_browser():
    global _pw, _browser, _page, _browser_lock
    _browser_lock = asyncio.Lock()
    proxy_display = _proxy["server"] if _proxy else "none"
    print(f"  Launching persistent Chromium browser (proxy: {proxy_display}) ...")
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless=_headless,
        proxy=_proxy,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--window-size=400,620",
        ],
    )
    session_file = str(_session_path) if (_session_path and _session_path.exists()) else None
    if session_file:
        print(f"  Loading saved session from {session_file}")
    ctx = await _browser.new_context(
        viewport={"width": CAPTCHA_W, "height": CAPTCHA_H},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        proxy=_proxy,
        storage_state=session_file,
    )
    await ctx.add_init_script(_STEALTH_SCRIPT)
    _page = await ctx.new_page()
    await _page.goto("about:blank")
    print("  Browser ready.")


async def _reset_page():
    """Navigate back to blank and uninstall any routes, ready for next solve."""
    try:
        await _page.unroute("**/*")
        await _page.goto("about:blank", wait_until="commit", timeout=10_000)
    except Exception:
        pass


async def _save_session_flow(out_path: Path):
    """
    Open a headed browser, navigate to Gmail, wait for the user to finish
    logging in, then persist all cookies + localStorage to out_path.
    """
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=False,
        proxy=_proxy,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )
    ctx = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        proxy=_proxy,
    )
    await ctx.add_init_script(_STEALTH_SCRIPT)
    page = await ctx.new_page()
    await page.goto("https://accounts.google.com/", timeout=30_000)

    print("\n  Browser opened — log in to your Google account, then come back here.")
    print("  Press Enter once you are fully logged in ...")
    # Block until the user presses Enter (run in executor so the event loop stays alive)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input, "")

    await ctx.storage_state(path=str(out_path))
    print(f"  Session saved to {out_path}")
    await browser.close()
    await pw.stop()


async def solve_captcha(pageurl: str) -> str | None:
    async with _browser_lock:
        page = _page

        # Install resource blocker fresh each request
        await page.route("**/*", _block_non_recaptcha)

        try:
            await page.goto(pageurl, timeout=30_000, wait_until="domcontentloaded")

            # Click the checkbox via frame_locator (works across browser engines)
            await page.wait_for_selector('iframe[src*="api2/anchor"]', timeout=10_000)
            await page.frame_locator('iframe[src*="api2/anchor"]') \
                      .locator('#recaptcha-anchor').click()
            await asyncio.sleep(random.uniform(1.5, 2.5))

            _zoomed = False

            for round_num in range(1, MAX_ROUNDS + 1):
                # Check if already solved
                token = await page.evaluate("""() => {
                    const el = document.querySelector(
                        '#g-recaptcha-response, textarea[name="g-recaptcha-response"]');
                    return el ? el.value : '';
                }""")
                if token:
                    print(f"  Solved in {round_num - 1} round(s).")
                    return token

                # Wait for challenge frame
                try:
                    bframe_el = await page.wait_for_selector(
                        'iframe[src*="api2/bframe"]', timeout=8_000)
                except Exception:
                    break
                frame_b = await bframe_el.content_frame()

                # Scroll instantly so the bframe sits flush at the top-left
                if not _zoomed:
                    box = await bframe_el.bounding_box()
                    if box:
                        await page.evaluate(
                            f"window.scrollTo({{top:{int(box['y'])},left:{int(box['x'])},behavior:'instant'}})"
                        )
                    _zoomed = True

                instr_el = await frame_b.query_selector('.rc-imageselect-instructions')
                if not instr_el:
                    await asyncio.sleep(1)
                    continue

                instruction = (await instr_el.text_content() or "").strip()
                category = extract_category(instruction)
                class_idx = category_to_index(category) if category else None

                print(f"  Round {round_num}: '{instruction}' -> {category} (idx={class_idx})")

                if class_idx is None or category in SKIP_CATEGORIES:
                    reason = "skipped category" if category in SKIP_CATEGORIES else "unknown category"
                    print(f"  {reason} ({category}) — reloading")
                    reload_btn = await frame_b.query_selector('#recaptcha-reload-button')
                    if reload_btn:
                        await reload_btn.click()
                    await asyncio.sleep(random.uniform(1.5, 2.5))
                    continue

                # Wait until every tile is loaded AND the CSS fade-in is complete.
                # The fade transition is on the <td>, not the <img>, so check td opacity.
                try:
                    await frame_b.wait_for_function("""() => {
                        const tiles = Array.from(
                            document.querySelectorAll('td.rc-imageselect-tile'));
                        if (tiles.length === 0) return false;
                        const stillLoading = tiles.some(td =>
                            td.classList.contains('rc-imageselect-tile--loading') ||
                            td.querySelector('.rc-imageselect-tile-loading') !== null
                        );
                        const imgsReady = Array.from(
                            document.querySelectorAll('td.rc-imageselect-tile img'))
                            .every(img => img.complete && img.naturalWidth > 0);
                        const tilesVisible = tiles.every(td =>
                            parseFloat(window.getComputedStyle(td).opacity) > 0.95
                        );
                        return !stillLoading && imgsReady && tilesVisible;
                    }""", timeout=15_000)
                except Exception:
                    pass
                # Hard floor: reCAPTCHA's fade animation is ~600 ms
                await asyncio.sleep(0.7)

                # One screenshot of the whole frame, then crop tiles in Python (no jitter)
                tiles = await frame_b.query_selector_all('td.rc-imageselect-tile')
                frame_png = await bframe_el.screenshot()
                frame_img = Image.open(BytesIO(frame_png))

                # Collect bounding boxes for all tiles
                tile_boxes = []
                for tile in tiles:
                    box = await tile.bounding_box()
                    tile_boxes.append(box)

                n_tiles   = len(tiles)
                is_4x4    = (n_tiles == 16)
                threshold = 0.1 if is_4x4 else CONF_THRESHOLD

                if is_4x4:
                    # 4×4 = one image split into 16 segments — use YOLO detection
                    valid_boxes = [b for b in tile_boxes if b]
                    yolo_scores = detect_4x4(frame_img, valid_boxes, category) if valid_boxes else None
                    if yolo_scores is not None:
                        # Remap to full tile list (some boxes may be missing)
                        confs = []
                        yolo_iter = iter(yolo_scores)
                        for b in tile_boxes:
                            confs.append(next(yolo_iter) if b else 0.0)
                        print(f"  4×4 YOLO scores: {[round(c,2) for c in confs]}")
                    else:
                        # YOLO not configured — fall back to per-tile classifier
                        confs = []
                        for i, (tile, box) in enumerate(zip(tiles, tile_boxes)):
                            if not box:
                                confs.append(0.0)
                                continue
                            x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
                            tile_img = frame_img.crop((x, y, x + w, y + h))
                            confs.append(float(classify(tile_img)[class_idx]))
                else:
                    # 3×3 or other — classify each tile independently
                    confs = []
                    for box in tile_boxes:
                        if not box:
                            confs.append(0.0)
                            continue
                        x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
                        tile_img = frame_img.crop((x, y, x + w, y + h))
                        confs.append(float(classify(tile_img)[class_idx]))

                # Overlay confidence scores on each tile
                overlay_data = [
                    {"index": i, "conf": c, "click": c >= threshold}
                    for i, c in enumerate(confs)
                ]
                await frame_b.evaluate("""(data) => {
                    document.querySelectorAll('.sv-overlay').forEach(e => e.remove());
                    const tiles = document.querySelectorAll('td.rc-imageselect-tile');
                    data.forEach(({index, conf, click}) => {
                        const td = tiles[index];
                        if (!td) return;
                        td.style.position = 'relative';
                        const ov = document.createElement('div');
                        ov.className = 'sv-overlay';
                        const pct = (conf * 100).toFixed(1);
                        const bg = click
                            ? 'rgba(0,210,0,0.55)'
                            : conf > 0.4
                                ? 'rgba(255,180,0,0.55)'
                                : 'rgba(210,0,0,0.45)';
                        ov.style.cssText = [
                            'position:absolute','inset:0','z-index:9999',
                            'pointer-events:none','display:flex',
                            'align-items:flex-end','justify-content:center',
                            'padding-bottom:4px',
                            `outline:5px solid ${bg}`,
                            'background:transparent',
                            'font:bold 12px/1 sans-serif',
                            `color:${bg.replace(/,[^,]+\\)/, ',1)')}`,
                            'text-shadow:0 0 3px #000'
                        ].join(';');
                        ov.textContent = pct + '%';
                        td.appendChild(ov);
                    });
                }""", overlay_data)

                n = len(tiles)
                above = [(i, c) for i, c in enumerate(confs) if c >= threshold]

                # Sanity check: if >60% of tiles are above threshold the model
                # is likely biased for this category — reload and try again
                if len(above) > max(1, int(n * 0.6)):
                    print(f"  Too many above threshold ({len(above)}/{n}) — reloading")
                    reload_btn = await frame_b.query_selector('#recaptcha-reload-button')
                    if reload_btn:
                        await reload_btn.click()
                    await asyncio.sleep(1.5)
                    continue

                # Skip tiles that are already selected (clicking them again would deselect)
                selected = await frame_b.evaluate("""() => {
                    return Array.from(document.querySelectorAll('td.rc-imageselect-tile'))
                        .map(td => td.classList.contains('rc-imageselect-tile--selected'));
                }""")
                to_click = [
                    tiles[i] for i, c in above
                    if not (i < len(selected) and selected[i])
                ]
                for i, c in enumerate(confs):
                    sel = i < len(selected) and selected[i]
                    label = 'SELECTED' if sel else ('CLICK' if c >= CONF_THRESHOLD else 'skip')
                    print(f"    tile {i}: {c:.2f} {label}")
                print(f"  Clicking {len(to_click)} / {n} tiles")

                await asyncio.sleep(random.uniform(0.8, 1.4))  # human "thinking" pause

                if to_click:
                    # Snapshot tile image URLs before clicking so we can detect
                    # replacements even if they finish loading before our check
                    srcs_before = await frame_b.evaluate("""() =>
                        Array.from(document.querySelectorAll('td.rc-imageselect-tile img'))
                            .map(img => img.src)
                    """)

                    for tile in to_click:
                        await tile.click()
                        await asyncio.sleep(random.uniform(0.4, 0.9))  # human-speed between clicks

                    # Poll for tile replacements for up to 3 s after the last click.
                    # Checks every 250 ms so we catch both immediate and delayed swaps.
                    new_tiles = False
                    poll_end = time.monotonic() + 3.0
                    await asyncio.sleep(0.2)  # minimum settling time
                    while time.monotonic() < poll_end:
                        loading = await frame_b.evaluate("""() =>
                            Array.from(document.querySelectorAll('td.rc-imageselect-tile')).some(td =>
                                td.classList.contains('rc-imageselect-tile--loading') ||
                                td.querySelector('.rc-imageselect-tile-loading') !== null
                            )
                        """)
                        srcs_now = await frame_b.evaluate("""() =>
                            Array.from(document.querySelectorAll('td.rc-imageselect-tile img'))
                                .map(img => img.src)
                        """)
                        instr_now = (await instr_el.text_content() or "").lower()
                        if (loading
                                or srcs_now != srcs_before
                                or "also check" in instr_now
                                or "new image" in instr_now):
                            new_tiles = True
                            print("  New tiles detected — waiting for full load before re-classifying")
                            break
                        await asyncio.sleep(0.25)

                    if new_tiles:
                        continue

                    verify_btn = await frame_b.query_selector('#recaptcha-verify-button')
                    if verify_btn:
                        await asyncio.sleep(random.uniform(0.5, 1.0))
                        await verify_btn.click()
                    await asyncio.sleep(random.uniform(2.0, 3.0))
                else:
                    # Try SKIP first (correct for "none present"), fall back to reload
                    skip_btn = await frame_b.query_selector('#recaptcha-skip-button, .rc-button-default')
                    reload_btn = await frame_b.query_selector('#recaptcha-reload-button')
                    if skip_btn:
                        print("  No matches — clicking SKIP")
                        await skip_btn.click()
                    elif reload_btn:
                        print("  No matches — reloading")
                        await reload_btn.click()
                    await asyncio.sleep(1.5)

        finally:
            await _reset_page()

    return None

app = Flask(__name__)
CORS(app)


@app.post("/solve")
def solve():
    data = request.get_json(silent=True) or {}
    pageurl = data.get("pageurl", "").strip()
    if not pageurl:
        return jsonify({"error": "pageurl required"}), 400

    t0 = time.time()
    print(f"\n[solve] {pageurl}")

    future = asyncio.run_coroutine_threadsafe(solve_captcha(pageurl), _browser_loop)
    try:
        token = future.result(timeout=180)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  -> error after {elapsed:.1f}s: {e}")
        return jsonify({"error": str(e)}), 500

    elapsed = time.time() - t0
    if token:
        print(f"  -> token obtained in {elapsed:.1f}s")
        print(f"  -> token: {token[:80]}...")
        return jsonify({"token": token})
    print(f"  -> failed after {elapsed:.1f}s")
    return jsonify({"error": "could not solve captcha"}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"})

def main():
    global CONF_THRESHOLD, _headless, _browser_loop, _proxy, _session_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=Path, default=Path("../extension/model.onnx"))
    parser.add_argument("--class-map", type=Path, default=Path("../extension/class_map.json"))
    parser.add_argument("--port",      type=int,  default=5000)
    parser.add_argument("--conf",      type=float, default=CONF_THRESHOLD)
    parser.add_argument("--headless",  action="store_true", help="Run browser headless")
    parser.add_argument("--proxy",     type=str,  default=None,
                        help="Proxy: host:port:user:pass  OR  http(s)://[user:pass@]host:port")
    parser.add_argument("--yolo",      type=Path,
                        default=Path("../data-collection/yolov8s-worldv2.pt"),
                        help="YOLOWorld model for 4×4 detection (omit to disable)")
    parser.add_argument("--session",   type=Path, default=Path("session.json"),
                        help="Path to saved browser session (cookies + storage)")
    parser.add_argument("--save-session", action="store_true",
                        help="Open a browser to log in to Google, save session, then exit")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    _headless      = args.headless
    _session_path  = args.session
    if args.proxy:
        raw = args.proxy.strip()
        parts = raw.split(":")
        # host:port:user:pass  (4-part format from proxy lists)
        if len(parts) == 4 and not raw.startswith("http") and not raw.startswith("socks"):
            host, port, user, passwd = parts
            _proxy = {
                "server":   f"http://{host}:{port}",
                "username": user,
                "password": passwd,
            }
        else:
            # Standard URL form: http://user:pass@host:port  or  http://host:port
            _proxy = {"server": raw}
            from urllib.parse import urlparse
            parsed = urlparse(raw)
            if parsed.username:
                _proxy["username"] = parsed.username
            if parsed.password:
                _proxy["password"] = parsed.password

    if args.save_session:
        print("=" * 50)
        print("  Save-session mode — launching login browser ...")
        asyncio.run(_save_session_flow(args.session))
        return

    print("=" * 50)
    load_model(args.model, args.class_map)
    if args.yolo and args.yolo.exists():
        load_yolo(args.yolo)
    else:
        print("  YOLO not found — 4×4 grids will use classifier fallback")

    # Start persistent event loop in a background daemon thread
    _browser_loop = asyncio.new_event_loop()
    t = threading.Thread(target=_run_event_loop, args=(_browser_loop,), daemon=True)
    t.start()

    # Launch the browser synchronously before accepting requests
    future = asyncio.run_coroutine_threadsafe(_launch_browser(), _browser_loop)
    future.result(timeout=60)

    print(f"  Listening on http://localhost:{args.port}")
    print("=" * 50)

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
