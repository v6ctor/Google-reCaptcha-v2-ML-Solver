"""
Manual image reviewer.

Shows images in a 4x4 grid. Click any image to mark it for rejection (red
border). Press Enter or click "Reject & Next" to move marked images to
rejected/<category>/ and advance to the next batch. Use arrow keys to navigate.

Usage:
    python review.py
    python review.py --category cars   # start on a specific category
    python review.py --cols 5          # 5-column grid (default 4)
    python review.py --size 140        # thumbnail size in pixels (default 150)
"""

import argparse
import shutil
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk

BASE_FOLDER = Path("images")
REJECTED_FOLDER = Path("rejected")


class ReviewApp:
    def __init__(self, root, start_category: str | None, cols: int, thumb: int):
        self.root = root
        self.root.title("Image Reviewer")
        self.root.configure(bg="#1e1e1e")

        self.cols = cols
        self.rows = cols  # square grid
        self.batch_size = cols * cols
        self.thumb = thumb

        self.images: list[Path] = []
        self.offset = 0
        self.selected: set[int] = set()   # indices within current batch
        self.photo_refs: list = []

        self._build_ui()
        self._load_categories(start_category)
        self._bind_keys()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#2d2d2d", pady=6)
        top.pack(fill=tk.X, padx=0)

        tk.Label(top, text="Category:", bg="#2d2d2d", fg="white").pack(side=tk.LEFT, padx=(12, 4))
        self.cat_var = tk.StringVar()
        self.cat_combo = ttk.Combobox(top, textvariable=self.cat_var, state="readonly", width=18)
        self.cat_combo.pack(side=tk.LEFT)
        self.cat_combo.bind("<<ComboboxSelected>>", lambda _: self._on_category_change())

        self.progress_label = tk.Label(top, text="", bg="#2d2d2d", fg="#aaaaaa", width=40, anchor="w")
        self.progress_label.pack(side=tk.LEFT, padx=16)

        self.rejected_label = tk.Label(top, text="", bg="#2d2d2d", fg="#e07070")
        self.rejected_label.pack(side=tk.RIGHT, padx=12)

        # ── image grid ───────────────────────────────────────────────────────
        self.grid_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.grid_frame.pack(padx=8, pady=8)

        self.cells: list[tuple[tk.Frame, tk.Label]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                outer = tk.Frame(self.grid_frame, bd=3, relief=tk.FLAT, bg="#1e1e1e")
                outer.grid(row=r, column=c, padx=2, pady=2)
                lbl = tk.Label(outer, bg="#1e1e1e", cursor="hand2")
                lbl.pack()
                idx = r * self.cols + c
                lbl.bind("<Button-1>", lambda _, i=idx: self._toggle(i))
                outer.bind("<Button-1>", lambda _, i=idx: self._toggle(i))
                self.cells.append((outer, lbl))

        # ── bottom bar ───────────────────────────────────────────────────────
        bot = tk.Frame(self.root, bg="#2d2d2d", pady=6)
        bot.pack(fill=tk.X)

        btn_cfg = dict(bg="#3c3c3c", fg="white", relief=tk.FLAT, padx=10, pady=4, cursor="hand2")

        tk.Button(bot, text="← Back", command=self._prev_batch, **btn_cfg).pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(bot, text="Keep All  →", command=self._keep_all, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bot, text="Reject Marked  ↵", bg="#7a2020", fg="white",
                  relief=tk.FLAT, padx=10, pady=4, cursor="hand2",
                  command=self._reject_and_next).pack(side=tk.LEFT, padx=4)

        hint = tk.Label(bot, text="Click = mark  |  ← → navigate  |  Enter = reject marked",
                        bg="#2d2d2d", fg="#666666")
        hint.pack(side=tk.RIGHT, padx=12)

    def _bind_keys(self):
        self.root.bind("<Right>", lambda _: self._keep_all())
        self.root.bind("<Left>", lambda _: self._prev_batch())
        self.root.bind("<Return>", lambda _: self._reject_and_next())
        self.root.bind("<space>", lambda _: self._keep_all())

    # ------------------------------------------------------------------ data

    def _load_categories(self, start: str | None):
        cats = sorted(d.name for d in BASE_FOLDER.iterdir() if d.is_dir()) if BASE_FOLDER.exists() else []
        self.cat_combo["values"] = cats
        initial = start if (start and start in cats) else (cats[0] if cats else "")
        if initial:
            self.cat_var.set(initial)
            self._load_images()

    def _on_category_change(self):
        self.offset = 0
        self.selected.clear()
        self._load_images()

    def _load_images(self):
        cat = self.cat_var.get()
        folder = BASE_FOLDER / cat
        self.images = sorted(folder.glob("*.png")) if folder.exists() else []
        self._render_batch()

    # ------------------------------------------------------------------ rendering

    def _render_batch(self):
        self.selected.clear()
        self.photo_refs = []
        batch = self.images[self.offset: self.offset + self.batch_size]

        cat = self.cat_var.get()
        total = len(self.images)
        end = min(self.offset + self.batch_size, total)
        start_n = self.offset + 1 if total else 0

        rejected_count = len(list((REJECTED_FOLDER / cat).glob("*.png"))) if (REJECTED_FOLDER / cat).exists() else 0

        self.progress_label.config(text=f"{cat}  |  {start_n}–{end} of {total} images")
        self.rejected_label.config(text=f"rejected: {rejected_count}")

        for idx, (frame, lbl) in enumerate(self.cells):
            frame.config(bg="#1e1e1e", relief=tk.FLAT)
            if idx < len(batch):
                img = Image.open(batch[idx]).resize((self.thumb, self.thumb), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.photo_refs.append(photo)
                lbl.config(image=photo, bg="#1e1e1e")
            else:
                lbl.config(image="", bg="#1e1e1e")
                self.photo_refs.append(None)

    def _toggle(self, idx: int):
        batch = self.images[self.offset: self.offset + self.batch_size]
        if idx >= len(batch):
            return
        frame, lbl = self.cells[idx]
        if idx in self.selected:
            self.selected.discard(idx)
            frame.config(bg="#1e1e1e", relief=tk.FLAT)
            lbl.config(bg="#1e1e1e")
        else:
            self.selected.add(idx)
            frame.config(bg="#cc2222", relief=tk.SUNKEN)
            lbl.config(bg="#cc2222")

    # ------------------------------------------------------------------ actions

    def _reject_and_next(self):
        if not self.selected:
            self._keep_all()
            return

        cat = self.cat_var.get()
        rejected_dir = REJECTED_FOLDER / cat
        rejected_dir.mkdir(parents=True, exist_ok=True)

        batch = self.images[self.offset: self.offset + self.batch_size]
        for idx in sorted(self.selected, reverse=True):
            if idx < len(batch):
                src = batch[idx]
                src.rename(rejected_dir / src.name)

        # Reload image list; advance past what we just reviewed
        next_offset = self.offset + self.batch_size - len(self.selected)
        self._load_images()
        self.offset = min(next_offset, max(0, len(self.images) - 1))
        self._render_batch()

    def _keep_all(self):
        self.offset = min(self.offset + self.batch_size, max(0, len(self.images) - 1))
        self.selected.clear()
        self._render_batch()

    def _prev_batch(self):
        self.offset = max(0, self.offset - self.batch_size)
        self.selected.clear()
        self._render_batch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--cols", type=int, default=4, help="Grid columns (and rows)")
    parser.add_argument("--size", type=int, default=150, help="Thumbnail size in px")
    args = parser.parse_args()

    root = tk.Tk()
    root.resizable(False, False)
    ReviewApp(root, args.category, args.cols, args.size)
    root.mainloop()


if __name__ == "__main__":
    main()
