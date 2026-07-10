#!/usr/bin/env python3
"""
digitize_cushman_calibrated.py  --  Roadmap #8 (human-anchor hardening):
turn Cushman, Sheketoff, Wharton & Carey (2013, Cognition) Fig. 3 into calibrated
numbers for the child developmental reference. ANALYSIS ONLY — no model inference.

HONEST DESIGN NOTE (read this):
  * The Y-AXIS CALIBRATION is automated and reliable: we detect each panel's plot
    frame (the solid black border) and map top border -> 1.0, bottom border -> 0.0.
  * The MARKER DETECTION is NOT reliably automatable on this figure: four overlapping
    series (Naughty/Punish x error bars) plus in-panel legends make naive color/shape
    detection grab error-bar tips, legend keys, and axis pixels. `--auto` runs that
    experimental detector and prints its (poor) output so you can see it fail.
  * The trustworthy numbers below (READS) are careful reads of each marker against the
    detected gridlines (calibrated eyeballing, ~±0.03-0.05). The overlay it produces
    (cushman_digitized_overlay.png) lets anyone verify every read sits on its marker.
    For a publication-grade lock-down, confirm in WebPlotDigitizer or request the raw
    per-age means from the Cushman lab.

Usage:
  python code/digitize_cushman_calibrated.py            # calibrate + overlay + CSV
  python code/digitize_cushman_calibrated.py --auto     # also run the (unreliable) detector

Outputs (dataset/human_reference/):
  cushman_digitized_overlay.png   verification overlay of the calibrated reads
  cushman_naughty_digitized.csv   all four panels, both series, per age
"""
import os, csv, argparse
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
HR   = os.path.join(ROOT, "dataset", "human_reference")
IMG  = os.path.join(HR, "cushman.png")

AGES = [4, 5, 6, 7, 8]

# ---------------------------------------------------------------------------
# Careful calibrated reads of Fig. 3 (proportion judged), against the gridlines.
# panel -> series -> {age: value}.  ~±0.03-0.05.  Verify with the overlay.
# ---------------------------------------------------------------------------
READS = {
    "a_attempted_first": {
        "Naughty": {4: 0.62, 5: 0.77, 6: 0.66, 7: 0.74, 8: 0.69},
        "Punish":  {4: 0.65, 5: 0.71, 6: 0.50, 7: 0.57, 8: 0.41},
    },
    "b_accidental_first": {
        "Naughty": {4: 0.61, 5: 0.30, 6: 0.33, 7: 0.07, 8: 0.06},
        "Punish":  {4: 0.63, 5: 0.54, 6: 0.54, 7: 0.28, 8: 0.21},
    },
    "c_attempted_second": {
        "Naughty": {4: 0.36, 5: 0.55, 6: 0.68, 7: 0.87, 8: 0.95},
        "Punish":  {4: 0.23, 5: 0.48, 6: 0.52, 7: 0.63, 8: 0.80},
    },
    "d_accidental_second": {
        "Naughty": {4: 0.55, 5: 0.15, 6: 0.19, 7: 0.18, 8: 0.22},
        "Punish":  {4: 0.50, 5: 0.31, 6: 0.35, 7: 0.27, 8: 0.44},
    },
}
PANEL_TITLE = {
    "a_attempted_first":   "(a) Attempted, first",
    "b_accidental_first":  "(b) Accidental, first",
    "c_attempted_second":  "(c) Attempted, second",
    "d_accidental_second": "(d) Accidental, second",
}
# which panel sits in which quadrant of the 2x2 image grid
PANEL_QUADRANT = {
    "a_attempted_first":   (0, 0),
    "b_accidental_first":  (0, 1),
    "c_attempted_second":  (1, 0),
    "d_accidental_second": (1, 1),
}
# marker x-position inset within the plot frame (age4 .. age8), tuned to the figure
X_INSET_LO, X_INSET_HI = 0.13, 0.93


def detect_frame(gray, x0, x1, y0, y1):
    """Find the solid black plot border inside a quadrant box -> (left,right,top,bot)."""
    sub = gray[y0:y1, x0:x1]
    h, w = sub.shape
    dark = sub < 110
    # horizontal border rows: long dark runs spanning most of the width
    row_frac = dark.sum(axis=1) / w
    hrows = np.where(row_frac > 0.55)[0]
    # vertical border cols
    col_frac = dark.sum(axis=0) / h
    vcols = np.where(col_frac > 0.55)[0]
    if len(hrows) < 2 or len(vcols) < 2:
        return None
    top, bot = hrows.min(), hrows.max()
    left, right = vcols.min(), vcols.max()
    return (x0 + left, x0 + right, y0 + top, y0 + bot)


def detect_gridlines(gray, left, right, top, bot):
    """Detect horizontal gridlines (incl. frame) within the frame -> sorted y pixels."""
    sub = gray[top:bot + 1, left:right + 1]
    w = sub.shape[1]
    darkish = sub < 160
    frac = darkish.sum(axis=1) / w
    ys = np.where(frac > 0.5)[0]
    # cluster adjacent rows
    lines, run = [], []
    for y in ys:
        if run and y - run[-1] > 3:
            lines.append(int(np.mean(run))); run = []
        run.append(y)
    if run:
        lines.append(int(np.mean(run)))
    return [top + y for y in lines]


def auto_markers(rgb, left, right, top, bot):
    """EXPERIMENTAL / UNRELIABLE color-based marker detector (see honest note)."""
    sub = rgb[top:bot + 1, left:right + 1].astype(int)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    black = (r < 70) & (g < 70) & (b < 70)
    grey = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r > 110) & (r < 185)
    H = bot - top
    out = {}
    for name, mask in [("Naughty", black), ("Punish", grey)]:
        vals = []
        for k in range(5):
            xc = (X_INSET_LO + (X_INSET_HI - X_INSET_LO) * k / 4) * (right - left)
            band = mask[:, max(0, int(xc) - 12):int(xc) + 12]
            ys = np.where(band.any(axis=1))[0]
            vals.append(round(1 - float(np.median(ys)) / H, 3) if len(ys) else float("nan"))
        out[name] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true",
                    help="also run the experimental (unreliable) color marker detector")
    a = ap.parse_args()
    if not os.path.exists(IMG):
        raise SystemExit(f"missing {IMG}")

    im = Image.open(IMG).convert("RGB")
    rgb = np.asarray(im)
    gray = rgb.mean(axis=2)
    H, W = gray.shape
    midx, midy = W // 2, H // 2
    quad_box = {(0, 0): (0, midx, 0, midy), (0, 1): (midx, W, 0, midy),
                (1, 0): (0, midx, midy, H), (1, 1): (midx, W, midy, H)}

    draw = ImageDraw.Draw(im)
    frames = {}
    # detect raw frames per panel
    raw = {}
    for panel, quad in PANEL_QUADRANT.items():
        raw[panel] = detect_frame(gray, *quad_box[quad])

    # Right-column panels sometimes miss the RIGHT vertical border (legend/thin line).
    # The top/bottom borders (which set the y-calibration) are reliable; mirror the
    # same-row left-column plot width to recover a degenerate right border.
    row_width = {}   # quadrant row -> left-column plot width
    for panel, (r, c) in PANEL_QUADRANT.items():
        if c == 0 and raw[panel]:
            l, rr, t, b = raw[panel]
            row_width[r] = rr - l
    for panel, (r, c) in PANEL_QUADRANT.items():
        fr = raw[panel]
        if fr and (fr[1] - fr[0]) < 200 and r in row_width:
            l, rr, t, b = fr
            raw[panel] = (l, l + row_width[r], t, b)

    print("=== Y-AXIS CALIBRATION (automated, reliable) ===")
    for panel in PANEL_QUADRANT:
        fr = raw[panel]
        if not fr:
            print(f"  {panel}: FRAME NOT FOUND"); continue
        left, right, top, bot = fr
        frames[panel] = fr
        grid = detect_gridlines(gray, left, right, top, bot)
        print(f"  {PANEL_TITLE[panel]:26} frame x[{left},{right}] y[{top},{bot}]  "
              f"{len(grid)} gridlines detected")
        draw.rectangle([left, top, right, bot], outline=(0, 150, 255), width=2)

    # ---- overlay curated reads (calibrated to the detected frame) ----
    def to_px(panel, age, val):
        left, right, top, bot = frames[panel]
        x = left + (X_INSET_LO + (X_INSET_HI - X_INSET_LO) * (age - 4) / 4) * (right - left)
        y = bot - val * (bot - top)   # top=1.0, bot=0.0
        return x, y

    SERIES_COLOR = {"Naughty": (220, 20, 60), "Punish": (255, 140, 0)}
    for panel in READS:
        if panel not in frames:
            continue
        for series, byage in READS[panel].items():
            col = SERIES_COLOR[series]
            for age, val in byage.items():
                x, y = to_px(panel, age, val)
                r = 9
                draw.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
    im.save(os.path.join(HR, "cushman_digitized_overlay.png"))
    print(f"\nwrote {os.path.relpath(os.path.join(HR,'cushman_digitized_overlay.png'), ROOT)}"
          "  (red=Naughty reads, orange=Punish reads, blue=detected frames)")

    # ---- optional experimental auto-detection ----
    if a.auto:
        print("\n=== EXPERIMENTAL AUTO MARKER DETECTION (UNRELIABLE — for the overlay only) ===")
        for panel in READS:
            if panel not in frames:
                continue
            det = auto_markers(rgb, *frames[panel])
            for s in ("Naughty", "Punish"):
                mono = "non-monotonic/incl-NaN" if any(np.isnan(det[s])) else "ok-ish"
                print(f"  {PANEL_TITLE[panel]:26} {s:8} auto={det[s]}  [{mono}]")
        print("  -> compare to READS; mismatches confirm the detector is not trustworthy here.")

    # ---- export calibrated reads CSV ----
    out_csv = os.path.join(HR, "cushman_naughty_digitized.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "condition", "presentation", "series", "age", "proportion",
                    "method", "source"])
        for panel, series_d in READS.items():
            cond = "attempted" if "attempted" in panel else "accidental"
            pres = "first" if panel.endswith("first") else "second"
            for series, byage in series_d.items():
                for age, val in byage.items():
                    w.writerow([panel, cond, pres, series, age, val,
                                "calibrated read (~±0.05)",
                                "Cushman Sheketoff Wharton & Carey 2013 Cognition Fig.3"])
    print(f"wrote {os.path.relpath(out_csv, ROOT)}")

    # ---- derive PROPOSED child developmental bands (construct-matched: Naughty) ----
    # Age -> band mapping for the human_reference developmental ladder.
    BANDS = {"child_4_5": [4, 5], "child_6_7": [6, 7], "child_8plus": [8]}

    def band_vals(att_by_age, acc_by_age):
        rows = []
        for band, ages in BANDS.items():
            acc = float(np.mean([acc_by_age[ag] for ag in ages]))
            att = float(np.mean([att_by_age[ag] for ag in ages]))
            rows.append((band, round(acc, 3), round(att, 3), round(att - acc, 3)))
        return rows

    variants = {
        "presented_first": (READS["a_attempted_first"]["Naughty"],
                            READS["b_accidental_first"]["Naughty"]),
        "first_second_avg": (
            {ag: (READS["a_attempted_first"]["Naughty"][ag]
                  + READS["c_attempted_second"]["Naughty"][ag]) / 2 for ag in AGES},
            {ag: (READS["b_accidental_first"]["Naughty"][ag]
                  + READS["d_accidental_second"]["Naughty"][ag]) / 2 for ag in AGES}),
    }
    bands_csv = os.path.join(HR, "cushman_child_bands_PROPOSED.csv")
    with open(bands_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "measure", "group", "accidental_norm_blame",
                    "attempted_norm_blame", "contrast_attempted_minus_accidental"])
        for variant, (att, acc) in variants.items():
            for band, a_acc, a_att, con in band_vals(att, acc):
                w.writerow([variant, "Naughty(wrongness)", band, a_acc, a_att, con])
    print(f"wrote {os.path.relpath(bands_csv, ROOT)}")
    print("\n=== PROPOSED child bands (Naughty=wrongness, construct-matched to adult anchor) ===")
    for variant, (att, acc) in variants.items():
        print(f"  [{variant}]")
        for band, a_acc, a_att, con in band_vals(att, acc):
            print(f"    {band:12} accidental={a_acc:.3f}  attempted={a_att:.3f}  "
                  f"contrast={con:+.3f}")
    print("  (existing human_reference.csv had child_4_5 -0.14, child_6_7 +0.15, "
          "child_8plus +0.46 from a naughty+punish/text mix — NOT overwritten.)")


if __name__ == "__main__":
    main()
