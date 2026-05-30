"""
reCAPTCHA image scraper — parallel mode with proxy support.

Usage:
    python scrape.py                        # single browser, no proxy
    python scrape.py --workers 10           # 10 parallel browsers (no proxies)
    python scrape.py --workers 20 --proxy-file proxies.txt
    python scrape.py --workers 5 --headed   # show browser windows (debug)

Proxy file format (one per line, any of these):
    192.168.1.1:8080
    192.168.1.1:8080:username:password
    http://192.168.1.1:8080
    http://username:password@192.168.1.1:8080
    socks5://192.168.1.1:1080
"""

import argparse
import asyncio
import re
import uuid
import random
from pathlib import Path
from io import BytesIO

import aiohttp
from PIL import Image
from playwright.async_api import async_playwright

USER_AGENTS = [
    # Chrome 121-124 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.185 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.112 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.106 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.82 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.185 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.106 Safari/537.36",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.2277.128",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1280, "height": 800},
    {"width": 1600, "height": 900},
]

LOCALES = ["en-US", "en-GB", "en-CA", "en-AU"]
TIMEZONES = ["America/New_York", "America/Chicago", "America/Los_Angeles", "Europe/London", "America/Toronto"]

# Injected before any page JS runs — patches the most common automation tells
STEALTH_SCRIPT = """
() => {
    // Hide webdriver flag
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

    // Fake a populated plugin list (headless has 0)
    const pluginData = [
        { name: 'PDF Viewer',        filename: 'internal-pdf-viewer',          description: 'Portable Document Format' },
        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: 'Portable Document Format' },
        { name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer',        description: 'Portable Document Format' },
        { name: 'Microsoft Edge PDF Viewer', filename: 'msedgepdf',            description: 'Portable Document Format' },
        { name: 'WebKit built-in PDF', filename: 'webkit-pdf',                 description: 'Portable Document Format' },
    ];
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const arr = pluginData.map(d => {
                const p = Object.create(Plugin.prototype);
                Object.defineProperty(p, 'name',        { value: d.name });
                Object.defineProperty(p, 'filename',    { value: d.filename });
                Object.defineProperty(p, 'description', { value: d.description });
                Object.defineProperty(p, 'length',      { value: 0 });
                return p;
            });
            Object.defineProperty(arr, 'item',    { value: (i) => arr[i] });
            Object.defineProperty(arr, 'namedItem', { value: (n) => arr.find(p => p.name === n) || null });
            return arr;
        }
    });

    // Languages
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

    // Add chrome object that headless is missing
    if (!window.chrome) {
        window.chrome = {
            app: { isInstalled: false, InstallState: {}, RunningState: {} },
            runtime: {
                OnInstalledReason: {},
                OnRestartRequiredReason: {},
                PlatformArch: {},
                PlatformOs: {},
                RequestUpdateCheckStatus: {},
                connect: () => {},
                sendMessage: () => {},
            },
        };
    }

    // Fix notification permissions check used by reCAPTCHA
    const _origQuery = window.navigator.permissions.query.bind(navigator.permissions);
    window.navigator.permissions.query = (params) =>
        params.name === 'notifications'
            ? Promise.resolve({ state: Notification.permission })
            : _origQuery(params);

    // Mask broken toString on functions (some detection checks this)
    const _toString = Function.prototype.toString;
    Function.prototype.toString = function() {
        if (this === window.navigator.permissions.query) return 'function query() { [native code] }';
        return _toString.call(this);
    };
}
"""

BASE_FOLDER = Path("images")
BASE_FOLDER.mkdir(exist_ok=True)

TARGET = 30000

CATEGORY_ALIASES = {
    "bus": "buses",
    "hydrant": "hydrants",
    "bicycle": "bicycles",
    "car": "cars",
    "motorcycle": "motorcycles",
    "crosswalk": "crosswalks",
    "bridge": "bridges",
    "stair": "stairs",
    "boat": "boats",
    "taxi": "taxis",
    "tractor": "tractors",
    "chimney": "chimneys",
    "light": "lights",
    "hill": "hills",
    "meter": "meters",
}

def parse_proxy(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # already a URL
    if line.startswith(("http", "socks")):
        return {"server": line}
    parts = line.split(":")
    if len(parts) == 2:
        return {"server": f"http://{line}"}
    if len(parts) == 4:
        ip, port, user, pw = parts
        return {"server": f"http://{ip}:{port}", "username": user, "password": pw}
    return {"server": f"http://{line}"}


def load_proxies(path: Path) -> list[dict]:
    if not path.exists():
        return []
    proxies = []
    for line in path.read_text().splitlines():
        p = parse_proxy(line)
        if p:
            proxies.append(p)
    print(f"Loaded {len(proxies)} proxies from {path}")
    return proxies

def normalize_category(raw: str) -> str:
    name = raw.lower().strip()
    return CATEGORY_ALIASES.get(name, name)


def extract_category(annotation: str) -> str:
    raw = (
        annotation
        .replace(" ", "_")
        .replace("/", "_")
        .split("Click")[0]
        .split("If")[0]
        .split("_")[-1]
    )
    return normalize_category(raw)


def count_images(category: str) -> int:
    total = len(list((BASE_FOLDER / category).glob("*.png"))) if (BASE_FOLDER / category).exists() else 0
    for alias, canonical in CATEGORY_ALIASES.items():
        if canonical == category and (BASE_FOLDER / alias).exists():
            total += len(list((BASE_FOLDER / alias).glob("*.png")))
    return total


def all_done(seen: set[str]) -> bool:
    return bool(seen) and all(count_images(c) >= TARGET for c in seen)


def print_progress(seen: set[str], worker_id: int):
    lines = [f"\n─── Progress (worker {worker_id}) ───────────────────────"]
    for cat in sorted(seen):
        n = count_images(cat)
        filled = min(n * 25 // TARGET, 25)
        bar = "#" * filled + "-" * (25 - filled)
        status = "DONE " if n >= TARGET else f"{n:>5}"
        lines.append(f"  {cat:<15} {status}/{TARGET}  [{bar}]")
    print("\n".join(lines))

def random_delay(lo=500, hi=1000):
    return random.uniform(lo / 1000, hi / 1000)


def parse_position(style: str):
    tm = re.search(r'top:\s*(-?\d+)%', style)
    lm = re.search(r'left:\s*(-?\d+)%', style)
    return int(tm.group(1)) if tm else 0, int(lm.group(1)) if lm else 0


async def download_tiles(session: aiohttp.ClientSession, url: str, folder: Path, tiles_info: list) -> int:
    async with session.get(url) as resp:
        if resp.status != 200:
            return 0
        data = await resp.read()
    full = Image.open(BytesIO(data)).convert("RGB")
    w, h = full.size
    grid = int(len(tiles_info) ** 0.5)
    tw, th = w // grid, h // grid
    saved = 0
    for t in tiles_info:
        top_pct, left_pct = t["position"]
        x = abs(left_pct) * tw // 100
        y = abs(top_pct) * th // 100
        tile = full.crop((x, y, x + tw, y + th))
        tile.save(folder / f"tile_{t['id']}_{uuid.uuid4().hex[:8]}.png", "PNG")
        saved += 1
    return saved

async def worker(worker_id: int, proxies: list[dict | None], headed: bool,
                 iteration_log: int, restart_every: int):
    seen: set[str] = set()

    async with async_playwright() as p:
        while True:
            # Pick a random proxy on every restart
            proxy = random.choice(proxies) if proxies else None
            proxy_label = proxy["server"] if proxy else "no proxy"
            print(f"[w{worker_id}] Starting — {proxy_label}")

            launch_args: dict = {"headless": not headed}
            if proxy:
                launch_args["proxy"] = proxy

            try:
                browser = await p.chromium.launch(**launch_args)
            except Exception as e:
                print(f"[w{worker_id}] Failed to launch browser: {e} — retrying with new proxy")
                await asyncio.sleep(2)
                continue

            async with aiohttp.ClientSession() as session:
                context = None
                try:
                    # Fresh context per session = new fingerprint each restart
                    ua = random.choice(USER_AGENTS)
                    context = await browser.new_context(
                        user_agent=ua,
                        viewport=random.choice(VIEWPORTS),
                        locale=random.choice(LOCALES),
                        timezone_id=random.choice(TIMEZONES),
                        java_script_enabled=True,
                    )
                    await context.add_init_script(STEALTH_SCRIPT)
                    page = await context.new_page()
                    await page.goto("https://www.google.com/recaptcha/api2/demo", timeout=30000)

                    iframe_anchor = await page.wait_for_selector('iframe[src*="api2/anchor"]', timeout=10000)
                    frame_a = await iframe_anchor.content_frame()
                    checkbox = await frame_a.wait_for_selector('#recaptcha-anchor', timeout=10000)
                    await checkbox.click()
                    await asyncio.sleep(random_delay(1000, 1500))

                    prev_urls: set[str] = set()
                    iteration = 0

                    while True:
                        iteration += 1

                        # Scheduled browser restart to free Chromium memory
                        if iteration > restart_every:
                            print(f"[w{worker_id}] Scheduled restart after {iteration} iterations")
                            break

                        try:
                            iframe_b = await page.wait_for_selector('iframe[src*="api2/bframe"]', timeout=8000)
                            frame_b = await iframe_b.content_frame()

                            instr = await frame_b.query_selector('.rc-imageselect-instructions')
                            annotation = (await instr.text_content()).strip()
                            category = extract_category(annotation)
                            seen.add(category)

                            current = count_images(category)

                            if current >= TARGET:
                                if all_done(seen):
                                    print(f"[w{worker_id}] All categories complete — exiting")
                                    await browser.close()
                                    return
                                reload = await frame_b.query_selector('#recaptcha-reload-button')
                                if reload:
                                    await reload.click()
                                    await asyncio.sleep(random_delay(1200, 1800))
                                continue

                            folder = BASE_FOLDER / category
                            folder.mkdir(exist_ok=True)

                            tiles_els = await frame_b.query_selector_all('td.rc-imageselect-tile')
                            if not tiles_els:
                                await asyncio.sleep(1)
                                continue

                            tiles_info = []
                            sprite_url = None
                            for el in tiles_els:
                                tid = await el.get_attribute('id')
                                img_el = await el.query_selector('img')
                                wrapper = await el.query_selector('.rc-image-tile-wrapper')
                                if img_el and wrapper:
                                    src = await img_el.get_attribute('src')
                                    style = await img_el.get_attribute('style') or ""
                                    if src:
                                        sprite_url = src
                                        tiles_info.append({"id": tid, "position": parse_position(style)})

                            if sprite_url and sprite_url not in prev_urls and tiles_info:
                                prev_urls.add(sprite_url)
                                saved = await download_tiles(session, sprite_url, folder, tiles_info)
                                print(f"[w{worker_id}] '{category}' {current + saved}/{TARGET} (+{saved})")
                            else:
                                print(f"[w{worker_id}] Skipping duplicate/empty grid")

                            if iteration % iteration_log == 0:
                                print_progress(seen, worker_id)

                            reload = await frame_b.query_selector('#recaptcha-reload-button')
                            if reload:
                                await reload.click()
                                await asyncio.sleep(random_delay(1500, 2000))
                            else:
                                await asyncio.sleep(random_delay(500, 800))

                        except Exception as e:
                            print(f"[w{worker_id}] Inner error: {e} — retrying with new proxy")
                            await asyncio.sleep(2)
                            break  # break inner loop → pick new random proxy → new browser

                except Exception as e:
                    print(f"[w{worker_id}] Page error: {e} — retrying with new proxy")
                    await asyncio.sleep(3)
                finally:
                    if context:
                        try:
                            await context.close()
                        except Exception:
                            pass

            try:
                await browser.close()
            except Exception:
                pass

async def main(n_workers: int, proxy_file: Path, headed: bool,
               iteration_log: int, restart_every: int):
    proxies = load_proxies(proxy_file)
    if proxies:
        random.shuffle(proxies)  # randomize starting order across the pool

    tasks = [
        asyncio.create_task(
            worker(i, proxies, headed and i == 0, iteration_log, restart_every)
        )
        for i in range(n_workers)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel browser instances (default: 1)")
    parser.add_argument("--proxy-file", type=Path, default=Path("proxies.txt"),
                        help="Path to proxy list file (default: proxies.txt)")
    parser.add_argument("--headed", action="store_true",
                        help="Show the first browser window (useful for debugging)")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Print progress summary every N iterations per worker")
    parser.add_argument("--restart-every", type=int, default=50,
                        help="Restart browser every N iterations to free memory (default: 50)")
    args = parser.parse_args()

    print(f"Starting {args.workers} worker(s), target={TARGET} per category, "
          f"restart every {args.restart_every} iterations")
    asyncio.run(main(args.workers, args.proxy_file, args.headed,
                     args.log_every, args.restart_every))
