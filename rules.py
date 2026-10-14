"""EAS 510 - Project 1: Rule functions for the Digital Forensics Apprentice.

Each rule compares one suspect image against one registered original and
returns a dict:

    {
        "rule": 1,
        "name": "Metadata",
        "fired": bool,          # True if the rule found supporting evidence
        "score": int,           # points awarded out of "out_of"
        "out_of": int,
        "note": "Size ratio 0.85",   # short human-readable metric
        "metric": 0.85,              # raw similarity measure in [0, 1]
    }

The points budgets are fixed by the assignment: Rule 1 = 30, Rule 2 = 30,
Rule 3 = 40. The detector sums them into a 0-100 confidence score.

These starter implementations are deliberately WEAK baselines (they hinge on
simple thresholds). Improving them is the assignment.

Reads are cached per path so one suspect against many originals does not
re-decode images needlessly.
"""

import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image

#: Rule names used by SimpleDetector.evaluate() (V1).
RULES = ("rule1_metadata", "rule2_histogram", "rule3_template")


@lru_cache(maxsize=256)
def _arr(path):
    return cv2.imread(path)


@lru_cache(maxsize=256)
def _hist(path):
    img = _arr(path)
    if img is None:
        return None
    hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
    hist_all = np.concatenate(hists).ravel().astype(np.float32)
    cv2.normalize(hist_all, hist_all)
    return hist_all


def _downscale(img, max_dim=512):
    """Fit an image into `max_dim` before expensive cv2 work."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@lru_cache(maxsize=256)
def _gray(path):
    img = _arr(path)
    if img is None:
        return None
    return _downscale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


@lru_cache(maxsize=256)
def _size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def rule1_metadata(target, input_path):
    """Compare image dimensions and file signatures."""
    out = {
        "rule": 1,
        "name": "Metadata",
        "fired": False,
        "score": 0,
        "out_of": 30,
        "note": "Metadata similarity 0.00",
        "metric": 0.0,
    }

    try:
        src_path = target["path"]
        src_w, src_h = _size(src_path) or (0, 0)
        in_w, in_h = _size(input_path) or (0, 0)

        if not all((src_w, src_h, in_w, in_h)):
            return out

        width_ratio = min(src_w, in_w) / max(src_w, in_w)
        height_ratio = min(src_h, in_h) / max(src_h, in_h)
        metric = (width_ratio + height_ratio) / 2

        def signature(path):
            with open(path, "rb") as f:
                return f.read(8)

        src_sig = signature(src_path)
        in_sig = signature(input_path)

        src_jpeg = src_sig.startswith(b"\\xff\\xd8\\xff")
        in_jpeg = in_sig.startswith(b"\\xff\\xd8\\xff")
        src_png = src_sig.startswith(b"\\x89PNG\\r\\n\\x1a\\n")
        in_png = in_sig.startswith(b"\\x89PNG\\r\\n\\x1a\\n")

        if (src_jpeg or src_png) and (in_jpeg or in_png):
            if src_jpeg != in_jpeg:
                metric -= 0.05

        metric = max(0.0, min(1.0, metric))
        out["metric"] = round(metric, 3)
        out["note"] = f"Metadata similarity {metric:.2f}"

        if metric >= 0.5:
            out["fired"] = True
            out["score"] = int(round(30 * metric))

    except (OSError, TypeError, ValueError):
        pass

    return out

def rule2_histogram(target, input_path):
    """Color histogram correlation. Robust to crops and spatial changes."""
    out = {"rule": 2, "name": "Histogram", "fired": False, "score": 0,
           "out_of": 30, "note": "Correlation 0.00", "metric": 0.0}
    try:
        hs, hi = _hist(target["path"]), _hist(input_path)
        if hs is None or hi is None:
            return out
        corr = float(cv2.compareHist(hs, hi, cv2.HISTCMP_CORREL))
        out["metric"] = round(max(0.0, min(1.0, corr)), 3)
        out["note"] = f"Correlation {out['metric']:.2f}"
        if out["metric"] >= 0.5:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out


def rule3_template(target, input_path):
    """Use ORB local features and geometric verification to detect crops."""
    out = {"rule": 3, "name": "Template", "fired": False, "score": 0,
           "out_of": 40, "note": "ORB matches 0, inliers 0", "metric": 0.0}

    try:
        src_g = _gray(target["path"])
        img_g = _gray(input_path)
        if src_g is None or img_g is None:
            return out

        orb = cv2.ORB_create(nfeatures=2500)
        kp1, des1 = orb.detectAndCompute(src_g, None)
        kp2, des2 = orb.detectAndCompute(img_g, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return out

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]

        if len(good) < 8:
            out["note"] = f"ORB matches {len(good)}, inliers 0"
            return out

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        metric = min(1.0, inliers / 25.0)
        out["metric"] = round(metric, 3)
        out["note"] = f"ORB matches {len(good)}, inliers {inliers}"

        if inliers >= 6 and len(good) >= 12:
            out["fired"] = True
            out["score"] = int(round(40 * metric))

    except Exception as exc:
        out["note"] = f"ORB error: {type(exc).__name__}"

    return out
