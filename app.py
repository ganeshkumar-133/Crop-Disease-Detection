from pathlib import Path
import json
import re
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cropguard.onnx"
LABELS_PATH = ROOT / "models" / "classes.json"

st.set_page_config(page_title="CropShield AI | Plant Health Scan", page_icon="🌿", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Manrope',sans-serif;}
.block-container {padding-top:2rem; max-width:1200px;}
.hero {background:linear-gradient(120deg,#102e26,#176b4b);color:#fff;border-radius:24px;padding:32px 34px;margin-bottom:22px;}
.eyebrow {font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;color:#b8f0d2;font-weight:700;}
.hero h1 {font-size:clamp(2rem,4vw,3.5rem);line-height:1.04;margin:.5rem 0 1rem;color:white;}
.hero p {color:#e0f4e8;max-width:760px;font-size:1.02rem;}
.metric {background:#f3f8f4;border:1px solid #e0ece3;border-radius:14px;padding:16px;min-height:100px;}
.metric-label {font-size:.75rem;text-transform:uppercase;letter-spacing:.09em;color:#52705d;}
.metric-value {font-family:'Manrope',sans-serif;font-size:1.2rem;font-weight:800;color:#173d2b;margin-top:8px;overflow-wrap:anywhere;}
.small-note {font-size:.88rem;color:#5f7466;}
.stButton>button {border-radius:12px;font-weight:700;}
</style>
<div class="hero">
  <div class="eyebrow">CropShield AI · Local plant health workspace</div>
  <h1>Spot the signs.<br>Protect the harvest.</h1>
  <p>Upload a clear leaf photo to check image quality and estimate which of the CropGuard model's supported plant-health classes best matches it. Runs locally on your computer.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing {LABELS_PATH.name}. Put it beside cropguard.onnx in the models folder.")
    import onnxruntime as ort
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    raw = json.loads(LABELS_PATH.read_text(encoding="utf-8-sig"))
    labels = parse_labels(raw)
    return session, labels

def parse_labels(raw):
    """Support list labels, index->label dictionaries, and label->index dictionaries."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        # Common formats: {"0": "Apple___healthy"} or {"Apple___healthy": 0}.
        if raw and all(str(k).isdigit() for k in raw.keys()):
            return [str(raw[k]) for k in sorted(raw.keys(), key=lambda k: int(k))]
        if raw and all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in raw.values()):
            return [str(k) for k, v in sorted(raw.items(), key=lambda item: int(item[1]))]
        # Some model cards wrap labels in a named property.
        for key in ("classes", "class_names", "labels", "idx_to_class"):
            if key in raw:
                return parse_labels(raw[key])
    raise ValueError("Unsupported classes.json format. Please use classes.json from the same CropGuard model repository.")

def analyze_image(image: Image.Image):
    """Detect color based on the share of genuinely chromatic pixels, not only the average."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb_i = rgb.astype(np.int16)
    channel_range = rgb_i.max(axis=2) - rgb_i.min(axis=2)
    brightness = rgb_i.max(axis=2)
    # White powder/sky/background can dominate average saturation. Count colored pixels instead.
    colored_pixels = (channel_range >= 20) & (brightness >= 35)
    color_pixel_percent = float(colored_pixels.mean() * 100.0)
    mean_channel_spread = float(channel_range.astype(np.float32).mean())
    is_color = color_pixel_percent >= 2.0

    gray = np.asarray(ImageOps.grayscale(image).resize((min(image.width, 512), min(image.height, 512))))
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        g = gray.astype(np.float32)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
               + g[1:-1, :-2] + g[1:-1, 2:])
        blur_score = float(lap.var())
    else:
        blur_score = 0.0
    sharpness = "Blurry" if blur_score < 45 else ("Slightly blurry" if blur_score < 120 else "Sharp")
    min_dim = min(image.width, image.height)
    if min_dim < 224:
        resolution_note = "Low resolution for this model; use a larger, clearer leaf photo if possible."
    elif min_dim < 400:
        resolution_note = "Usable, but a higher-resolution photo may preserve more leaf detail."
    else:
        resolution_note = "Resolution looks suitable for a close-up model input."
    return {
        "image_type": "Color image" if is_color else "Black and white image",
        "color_pixel_percent": color_pixel_percent,
        "channel_spread": mean_channel_spread,
        "blur_score": blur_score,
        "sharpness": sharpness,
        "width": image.width,
        "height": image.height,
        "resolution_note": resolution_note,
    }

def format_label(label):
    text = str(label).replace("___", " — ").replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)

def predict_disease(image, session, labels):
    # CropGuard model-card preprocessing: shortest side 256, center crop 224, RGB,
    # [0,1] scale, ImageNet normalization, NCHW. Keep in sync with the model card.
    image = image.convert("RGB")
    w, h = image.size
    scale = 256.0 / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = np.transpose((array - mean) / std, (2, 0, 1))[None, ...].astype(np.float32)
    input_meta = session.get_inputs()[0]
    outputs = session.run(None, {input_meta.name: batch})
    scores = np.asarray(outputs[0]).reshape(-1).astype(np.float64)
    if len(labels) != len(scores):
        raise ValueError(f"Model returned {len(scores)} class scores but classes.json contains {len(labels)} labels. Download both files from the same model repository.")
    # Temperature scaling is recommended by this model's model card. Scores are still
    # model scores, not a guarantee of real-world diagnostic probability.
    scores = scores / 0.591
    scores -= np.max(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores))
    order = np.argsort(probs)[::-1]
    ranked = [(labels[int(i)], float(probs[int(i)])) for i in order[:3]]
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], top_score, ranked, top_score - second_score

try:
    model, labels = load_model()
    model_error = None
except Exception as exc:
    model, labels, model_error = None, None, str(exc)

left, right = st.columns([1.1, .9], gap="large")
with left:
    st.subheader("1 · Upload a plant image")
    uploaded = st.file_uploader("Choose a leaf or crop image", type=["jpg", "jpeg", "png", "webp", "bmp"], help="Best results usually come from a well-lit close-up of one leaf, filling most of the frame.")
    image_kind = st.radio("Image source", ["Close-up leaf photo", "Drone / field image"], horizontal=True)
    st.caption("Drone / field images are experimental: the model was trained on close-up leaf photographs, not aerial imagery.")
with right:
    st.subheader("2 · Scan results")
    if uploaded is None:
        st.info("Upload an image to view image-quality checks and a model prediction.")
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
            details = analyze_image(image)
            st.image(image, caption=f"{uploaded.name} · {details['width']} × {details['height']}", use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric"><div class="metric-label">Image format</div><div class="metric-value">{details["image_type"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric"><div class="metric-label">Image sharpness</div><div class="metric-value">{details["sharpness"]}</div></div>', unsafe_allow_html=True)
            st.caption(f"Colored pixels: {details['color_pixel_percent']:.1f}% · Average channel difference: {details['channel_spread']:.1f} · Edge-detail score: {details['blur_score']:.1f}")
            if details["color_pixel_percent"] < 2.0:
                st.info("Very little color was detected. If this photo should be in color, check that the original file is not grayscale or heavily washed out.")
            if details["width"] < 400 or details["height"] < 400:
                st.warning(details["resolution_note"])
            else:
                st.caption(details["resolution_note"])
            st.divider()
            if model_error:
                st.error(f"Model setup issue: {model_error}")
            elif model is None:
                st.warning("The ONNX disease model is not installed. Put cropguard.onnx and classes.json in the project's models folder.")
            else:
                try:
                    name, score, ranking, margin = predict_disease(image, model, labels)
                    nice_name = format_label(name)
                    st.markdown("#### Model's best match")
                    st.markdown(f"### {nice_name}")
                    st.metric("Top-class model score", f"{score:.1%}", help="Softmax score after the model-card temperature adjustment. This is not a verified real-world probability or diagnostic confidence.")
                    st.progress(min(max(score, 0.0), 1.0))
                    if score < 0.55 or margin < 0.15:
                        st.warning("Low certainty: the leading class is not clearly separated from alternatives. Try a sharper close-up image, and do not use this prediction alone to decide treatment.")
                    else:
                        st.info("The model favors this class among its supported labels. This score does not guarantee the crop or disease was recognized correctly.")
                    with st.expander("Compare the top 3 model scores", expanded=True):
                        for label, value in ranking:
                            st.write(f"**{format_label(label)}** — {value:.1%}")
                            st.progress(min(max(value, 0.0), 1.0))
                    if image_kind == "Drone / field image":
                        st.warning("This model is not validated for aerial field images. The result may be unreliable even if the displayed score is high.")
                except Exception as exc:
                    st.error(f"The model could not analyze this image: {exc}")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Could not open this image: {exc}")

st.divider()
with st.expander("Model coverage and limitations"):
    st.markdown("""
    - This app uses the free CropGuard ONNX model and its matching `classes.json` file.
    - The model lists 38 classes across 14 crops. It does not cover every crop or disease; an unsupported crop may still be assigned to one of the known classes.
    - Training images are close-up leaves photographed against relatively plain backgrounds. Whole-field drone images are not validated.
    - A model score is not the same as proven diagnostic accuracy or a real-world probability. Image conditions and crops outside the training classes can cause wrong predictions.
    - Use the output as a student-project aid, not as a substitute for an agricultural expert.
    - The upstream model is licensed CC BY-SA 4.0. Keep attribution and share-alike terms when redistributing adaptations.
    """)
st.caption("Local CPU inference · No paid API required · After packages and model files are installed, prediction runs on this computer.")
