
from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError


# ============================================================
# CROPShield: CANDIDATE DISEASE-REGION LOCALIZATION
# Uses the existing four-class soybean ONNX classifier.
# Does not modify the original 38-class CropGuard model.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cropshield_soybean.onnx"
CLASSES_PATH = ROOT / "models" / "soybean_classes.json"

TILE_SIZE = 224
TILE_STEP = 112
MAX_IMAGE_SIDE = 1200
MAX_TILES = 100
MAX_BOXES = 20

# These are heuristic thresholds, not medically or agriculturally
# validated detection thresholds.
MIN_DISEASE_SCORE = 0.65
MIN_SCORE_MARGIN = 0.10

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

st.set_page_config(
    page_title="Disease Localization | CropShield",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 CropShield — Plant Disease Localization")

st.write(
    "Scan overlapping regions of a soybean image and highlight "
    "candidate regions classified as Mosaic, Rust, or Semilooper/Pest."
)

st.warning(
    "Experimental localization: this is a sliding-window classifier, "
    "not a trained object detector. Boxes identify image regions whose "
    "classification scores pass heuristic thresholds. They do not prove "
    "that a disease or pest is present in that exact region."
)

st.info(
    "This page uses cropshield_soybean.onnx and soybean_classes.json. "
    "The original 38-class CropGuard model and existing pages are unchanged."
)


# ------------------------------------------------------------
# Load the separate soybean ONNX classifier
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(f"Class labels not found: {CLASSES_PATH}")

    with CLASSES_PATH.open("r", encoding="utf-8-sig") as f:
        classes = json.load(f)

    if not isinstance(classes, list) or len(classes) != 4:
        raise ValueError(
            "soybean_classes.json must contain exactly four class labels."
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    model_input = session.get_inputs()[0]
    shape = model_input.shape

    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(
            f"Expected NHWC input [batch, height, width, 3], got {shape}."
        )

    height = shape[1] if isinstance(shape[1], int) else 224
    width = shape[2] if isinstance(shape[2], int) else 224

    if width <= 0 or height <= 0:
        raise ValueError("The model input dimensions are invalid.")

    return session, classes, model_input.name, width, height


def get_probabilities(raw_scores):
    """Convert classifier output to a usable score distribution."""
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid numerical scores.")

    # Keep valid probabilities; otherwise interpret the output as logits.
    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(float(scores.sum()), 1.0, atol=0.02)
    ):
        return scores

    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    total = float(exp_scores.sum())

    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not normalize the model output.")

    return exp_scores / total


def predict_tile(tile, session, classes, input_name, width, height):
    tile = tile.convert("RGB").resize((width, height))

    # The model's preprocessing is embedded in the ONNX graph.
    array = np.asarray(tile, dtype=np.float32)[None, ...]

    outputs = session.run(None, {input_name: array})
    probabilities = get_probabilities(outputs[0][0])

    if len(probabilities) != len(classes):
        raise ValueError("Model scores do not match the class labels.")

    best_index = int(np.argmax(probabilities))

    return classes[best_index], float(probabilities[best_index]), probabilities


# ------------------------------------------------------------
# Image resizing and overlapping sliding-window coordinates
# ------------------------------------------------------------

def resize_for_scanning(image):
    original_width, original_height = image.size
    scale = min(
        1.0,
        MAX_IMAGE_SIDE / max(original_width, original_height),
    )

    preview_width = max(1, round(original_width * scale))
    preview_height = max(1, round(original_height * scale))

    preview = image.resize((preview_width, preview_height))

    return preview, scale, original_width, original_height


def positions(length, tile_size, step):
    if length <= tile_size:
        return [0]

    result = list(range(0, length - tile_size + 1, step))

    # Always include the far edge of the image.
    last = length - tile_size
    if result[-1] != last:
        result.append(last)

    return result


def generate_tiles(image):
    width, height = image.size
    tile_width = min(TILE_SIZE, width)
    tile_height = min(TILE_SIZE, height)

    xs = positions(width, tile_width, TILE_STEP)
    ys = positions(height, tile_height, TILE_STEP)

    coordinates = [
        (x, y, min(x + tile_width, width), min(y + tile_height, height))
        for y in ys
        for x in xs
    ]

    if len(coordinates) > MAX_TILES:
        # Sample evenly across the tile list to limit CPU work.
        indexes = np.linspace(
            0,
            len(coordinates) - 1,
            MAX_TILES,
            dtype=int,
        )
        coordinates = [coordinates[i] for i in indexes]

    return coordinates


# ------------------------------------------------------------
# Filter overlapping candidate boxes with non-maximum suppression
# ------------------------------------------------------------

def box_iou(a, b):
    left = max(a["box"][0], b["box"][0])
    top = max(a["box"][1], b["box"][1])
    right = min(a["box"][2], b["box"][2])
    bottom = min(a["box"][3], b["box"][3])

    intersection = max(0, right - left) * max(0, bottom - top)

    area_a = (
        max(0, a["box"][2] - a["box"][0])
        * max(0, a["box"][3] - a["box"][1])
    )
    area_b = (
        max(0, b["box"][2] - b["box"][0])
        * max(0, b["box"][3] - b["box"][1])
    )

    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def suppress_overlaps(candidates, iou_threshold=0.35):
    kept = []

    # Suppress overlaps within each predicted class, not across classes.
    for class_name in sorted({c["class"] for c in candidates}):
        group = [
            c for c in candidates
            if c["class"] == class_name
        ]
        group.sort(key=lambda item: item["score"], reverse=True)

        while group:
            best = group.pop(0)
            kept.append(best)

            group = [
                item for item in group
                if box_iou(best, item) < iou_threshold
            ]

    kept.sort(key=lambda item: item["score"], reverse=True)
    return kept[:MAX_BOXES]


# ------------------------------------------------------------
# Scan the image and produce candidate boxes
# ------------------------------------------------------------

def localize_image(image, session, classes, input_name, width, height):
    preview, scale, original_width, original_height = resize_for_scanning(
        image.convert("RGB")
    )

    tiles = generate_tiles(preview)
    healthy_labels = {
        label.strip().lower()
        for label in classes
        if "healthy" in label.strip().lower()
    }

    candidates = []
    tile_results = []

    progress = st.progress(0, text="Scanning image regions...")

    for index, box in enumerate(tiles):
        tile = preview.crop(box)

        predicted, score, probabilities = predict_tile(
            tile,
            session,
            classes,
            input_name,
            width,
            height,
        )

        healthy_score = max(
            (
                float(probabilities[i])
                for i, label in enumerate(classes)
                if label.strip().lower() in healthy_labels
            ),
            default=0.0,
        )

        tile_results.append({
            "box": box,
            "class": predicted,
            "score": score,
        })

        # Only flag non-healthy classes that pass both heuristics.
        # These thresholds do not establish diagnostic validity.
        if (
            predicted.strip().lower() not in healthy_labels
            and score >= MIN_DISEASE_SCORE
            and score - healthy_score >= MIN_SCORE_MARGIN
        ):
            candidates.append({
                "box": box,
                "class": predicted,
                "score": score,
            })

        progress.progress(
            (index + 1) / len(tiles),
            text=f"Scanning region {index + 1} of {len(tiles)}...",
        )

    progress.empty()

    detections = suppress_overlaps(candidates)

    # Draw on the resized preview for display.
    annotated = preview.copy()
    draw = ImageDraw.Draw(annotated)

    palette = {
        "mosaic": "#FF4136",
        "rust": "#FF851B",
        "semilooper_pest": "#B10DC9",
    }

    for number, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        label_key = detection["class"].strip().lower().replace(" ", "_")
        color = palette.get(label_key, "#FF4136")

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=4,
        )

        label_text = (
            f"{number}. {detection['class']} "
            f"{detection['score']:.0%}"
        )

        text_box = draw.textbbox((x1, y1), label_text)
        text_height = text_box[3] - text_box[1]

        label_top = max(0, y1 - text_height - 6)

        draw.rectangle(
            (x1, label_top, x1 + (text_box[2] - text_box[0]) + 8,
             label_top + text_height + 6),
            fill=color,
        )
        draw.text(
            (x1 + 4, label_top + 3),
            label_text,
            fill="white",
        )

    # Convert displayed preview coordinates to original-image coordinates.
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        detection["original_box"] = (
            round(x1 / scale),
            round(y1 / scale),
            round(x2 / scale),
            round(y2 / scale),
        )

    return (
        annotated,
        detections,
        len(tiles),
        original_width,
        original_height,
        tile_results,
    )


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

try:
    session, classes, input_name, model_width, model_height = load_model()
except Exception as exc:
    st.error(f"Could not load the soybean model: {exc}")
    st.stop()

st.success("Soybean ONNX classifier loaded on CPU.")

st.caption(
    f"Model input: {model_width} × {model_height} | "
    f"Maximum scanned regions per image: {MAX_TILES}"
)

uploads = st.file_uploader(
    "Upload soybean images",
    type=IMAGE_TYPES,
    accept_multiple_files=True,
)

if uploads:
    for uploaded in uploads:
        st.divider()
        st.subheader(uploaded.name)

        try:
            original = Image.open(uploaded).convert("RGB")

            with st.spinner("Scanning overlapping image regions..."):
                (
                    annotated,
                    detections,
                    tile_count,
                    original_width,
                    original_height,
                    tile_results,
                ) = localize_image(
                    original,
                    session,
                    classes,
                    input_name,
                    model_width,
                    model_height,
                )

            left, right = st.columns([1.2, 1])

            with left:
                st.image(
                    annotated,
                    caption="Candidate-region boxes — not confirmed lesions",
                    use_container_width=True,
                )

            with right:
                st.metric("Image regions scanned", tile_count)
                st.metric("Candidate boxes", len(detections))

                if detections:
                    st.warning(
                        "Candidate regions were found. Review the original "
                        "image; these boxes are not validated detections."
                    )

                    rows = []
                    for number, item in enumerate(detections, start=1):
                        x1, y1, x2, y2 = item["original_box"]
                        rows.append({
                            "Box": number,
                            "Predicted class": item["class"],
                            "Model score": f"{item['score']:.2%}",
                            "X1": x1,
                            "Y1": y1,
                            "X2": x2,
                            "Y2": y2,
                        })

                    st.dataframe(rows, use_container_width=True)

                    import csv
                    import io

                    output = io.StringIO()
                    writer = csv.DictWriter(
                        output,
                        fieldnames=list(rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(rows)

                    st.download_button(
                        "Download candidate boxes (CSV)",
                        data=output.getvalue(),
                        file_name=f"{Path(uploaded.name).stem}_boxes.csv",
                        mime="text/csv",
                        key=f"csv_{uploaded.name}",
                    )
                else:
                    st.info(
                        "No regions passed the configured heuristic "
                        "thresholds. This does not prove the image is healthy."
                    )

            st.caption(
                f"Original image: {original_width} × {original_height} pixels. "
                "Coordinates in the table refer to the original image."
            )

        except (UnidentifiedImageError, OSError):
            st.error(f"Could not read image: {uploaded.name}")
        except Exception as exc:
            st.error(f"Could not analyze {uploaded.name}: {exc}")

else:
    st.info("Upload an image to scan it for candidate disease regions.")

st.divider()
st.subheader("How this prototype works")

st.markdown(
    """
    1. Resize the uploaded image for faster processing.
    2. Scan overlapping regions using the existing soybean classifier.
    3. Flag regions where a non-healthy class score passes two thresholds.
    4. Suppress highly overlapping boxes for the same predicted class.
    5. Display the candidate boxes and allow the coordinates to be exported.

    **Known limitation:** the classifier was not trained with bounding-box
    annotations. Its scores and boxes should not be used alone to diagnose
    crops or decide on pesticide treatment. A dedicated detector trained
    with annotated images is needed for reliable plant-level localization.
    """
)
