"""EAS 510 Assignment 1 - Phase 2 rules."""

import cv2
import numpy as np
import rules

RULES = (
    "rule1_metadata_v2",
    "rule2_histogram_v2",
    "rule3_template_v2",
    "rule4_edges",
)


def _rescore(result, points):
    """Keep the original rule's evidence and adjust its point budget."""
    result = result.copy()
    result["out_of"] = points
    result["score"] = (
        int(round(points * result["metric"])) if result["fired"] else 0
    )
    return result


def rule1_metadata_v2(target, input_path):
    return _rescore(rules.rule1_metadata(target, input_path), 25)


def rule2_histogram_v2(target, input_path):
    return _rescore(rules.rule2_histogram(target, input_path), 25)


def rule3_template_v2(target, input_path):
    return _rescore(rules.rule3_template(target, input_path), 40)


def rule4_edges(target, input_path):
    """Compare Canny edge structure to help with geometric image changes."""
    out = {
        "rule": 4,
        "name": "Edges",
        "fired": False,
        "score": 0,
        "out_of": 10,
        "note": "Edge match 0.00",
        "metric": 0.0,
    }

    try:
        original = cv2.imread(target["path"], cv2.IMREAD_GRAYSCALE)
        suspect = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if original is None or suspect is None:
            return out

        def prepare(img, max_dim=512):
            h, w = img.shape
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(
                    img,
                    (max(1, int(w * scale)), max(1, int(h * scale)))
                )
            return cv2.Canny(img, 60, 160)

        original = prepare(original)
        suspect = prepare(suspect)

        # Template matching requires the suspect to fit inside the original.
        if (suspect.shape[0] > original.shape[0]
                or suspect.shape[1] > original.shape[1]):
            scale = min(
                original.shape[0] / suspect.shape[0],
                original.shape[1] / suspect.shape[1]
            )
            suspect = cv2.resize(
                suspect,
                (
                    max(1, int(suspect.shape[1] * scale)),
                    max(1, int(suspect.shape[0] * scale))
                )
            )

        result = cv2.matchTemplate(
            original, suspect, cv2.TM_CCOEFF_NORMED
        )
        metric = float(result.max())

        # Ignore invalid or negative correlations.
        metric = max(0.0, min(1.0, metric))
        out["metric"] = round(metric, 3)
        out["note"] = f"Edge match {metric:.2f}"

        if metric >= 0.4:
            out["fired"] = True
            out["score"] = int(round(10 * metric))

    except Exception:
        pass

    return out
