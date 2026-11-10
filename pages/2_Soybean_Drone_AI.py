
from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "cropshield_soybean.onnx"
CLASSES_PATH = ROOT / "models" / "soybean_classes.json"

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
CONFIDENCE_THRESHOLD = 0.60


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Soybean Drone AI | CropShield",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 CropShield — Soybean Drone AI")

st.write(
    "Upload soybean leaf images captured by a drone or camera "
    "to classify them into one of four categories."
)

st.info(
    "This page uses a separate four-class soybean model. "
    "Your original 38-class CropGuard model remains unchanged."
)

st.warning(
    "Research prototype: predictions are estimates, not confirmed "
    "plant diagnoses. Verify results using independent field images."
)


# --------------------------------------------------
# 3. LOAD ONNX MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_soybean_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Soybean model not found:\n{MODEL_PATH}"
        )

    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(
            f"Soybean class labels not found:\n{CLASSES_PATH}"
        )

    with CLASSES_PATH.open("r", encoding="utf-8-sig") as file:
        classes = json.load(file)

    if not isinstance(classes, list) or len(classes) != 4:
        raise ValueError(
            "soybean_classes.json must contain a list of four class labels."
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    model_input = session.get_inputs()[0]
    shape = model_input.shape

    # This model expects NHWC input: [batch, height, width, channels].
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(
            f"Unexpected model input shape: {shape}. "
            "Expected [batch, height, width, 3]."
        )

    height = shape[1] if isinstance(shape[1], int) else 224
    width = shape[2] if isinstance(shape[2], int) else 224

    return (
        session,
        classes,
        model_input.name,
        width,
        height,
    )


# --------------------------------------------------
# 4. IMAGE PREPROCESSING AND PREDICTION
# --------------------------------------------------

def predict_image(
    image,
    session,
    classes,
    input_name,
    width,
    height,
):
    # Convert the image to RGB and resize it to model dimensions.
    image = image.convert("RGB").resize((width, height))

    # NHWC layout: [batch, height, width, RGB channels].
    # The model contains its own preprocessing.
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )[None, ...]

    outputs = session.run(
        None,
        {input_name: image_array},
    )

    scores = np.asarray(outputs[0])[0].reshape(-1)

    if len(scores) != len(classes):
        raise ValueError(
            f"Model returned {len(scores)} scores, "
            f"but {len(classes)} class labels were found."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid numerical scores.")

    # If the model returns logits, convert them to probabilities.
    # If it already returns probabilities, keep them unchanged.
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(np.sum(scores), 1.0, atol=0.02)
    ):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)
    else:
        probabilities = scores

    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])

    rankings = [
        (
            classes[int(index)],
            float(probabilities[int(index)]),
        )
        for index in order
    ]

    return (
        classes[best_index],
        float(probabilities[best_index]),
        rankings,
    )


# --------------------------------------------------
# 5. LOAD THE MODEL SAFELY
# --------------------------------------------------

try:
    (
        session,
        classes,
        input_name,
        model_width,
        model_height,
    ) = load_soybean_model()

except Exception as exc:
    st.error(f"Unable to load the soybean model: {exc}")

    st.markdown(
        """
        **Check these files exist:**

        - `models/cropshield_soybean.onnx`
        - `models/soybean_classes.json`

        Also check that ONNX Runtime is installed.
        """
    )

    st.stop()


st.success(
    f"Soybean ONNX model loaded successfully on the CPU "
    f"({model_width} × {model_height} input)."
)


# --------------------------------------------------
# 6. IMAGE UPLOAD
# --------------------------------------------------

st.header("📷 Upload soybean images")

uploads = st.file_uploader(
    "Choose one or more soybean leaf images",
    type=IMAGE_TYPES,
    accept_multiple_files=True,
    help=(
        "Use clear, well-lit images with visible leaf symptoms. "
        "Close-up images generally work better than wide aerial images."
    ),
)


# --------------------------------------------------
# 7. ANALYZE UPLOADED IMAGES
# --------------------------------------------------

if uploads:
    for uploaded in uploads:
        st.divider()
        st.subheader(f"🖼️ {uploaded.name}")

        try:
            image = Image.open(uploaded).convert("RGB")

            image_column, result_column = st.columns(
                [1, 1],
                gap="large",
            )

            with image_column:
                st.image(
                    image,
                    caption="Uploaded soybean image",
                    use_container_width=True,
                )

                st.caption(
                    f"Original image size: "
                    f"{image.width} × {image.height} pixels"
                )

            with st.spinner("Analyzing image with ONNX Runtime..."):
                (
                    predicted_class,
                    confidence,
                    rankings,
                ) = predict_image(
                    image,
                    session,
                    classes,
                    input_name,
                    model_width,
                    model_height,
                )

            with result_column:
                st.metric(
                    "Predicted category",
                    predicted_class,
                )

                st.metric(
                    "Top model score",
                    f"{confidence:.1%}",
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        "Low model score. Treat this result as uncertain "
                        "and review the image manually."
                    )
                else:
                    st.info(
                        "The model produced a higher top score. "
                        "This does not guarantee a correct classification."
                    )

                st.markdown("#### Class score ranking")

                for class_name, score in rankings:
                    st.write(f"**{class_name}** — {score:.2%}")

                    st.progress(
                        max(0.0, min(1.0, score))
                    )

        except (UnidentifiedImageError, OSError):
            st.error(
                f"Could not read '{uploaded.name}'. "
                "Please upload a valid image."
            )

        except Exception as exc:
            st.error(
                f"Prediction failed for '{uploaded.name}': {exc}"
            )


# --------------------------------------------------
# 8. DRONE WORKFLOW INFORMATION
# --------------------------------------------------

else:
    st.info(
        "Upload a soybean image to begin classification."
    )

st.divider()

st.header("🚁 Drone workflow")

st.write(
    """
    1. Capture soybean field images using a drone camera.
    2. Select clear leaf images or suitable cropped regions.
    3. Upload images to this page.
    4. Review the predicted category and model scores.
    5. Verify uncertain predictions before taking crop-management action.
    """
)

st.warning(
    "This prototype classifies each uploaded image as a whole. "
    "It does not yet detect individual plants, draw disease bounding boxes, "
    "or automatically process live drone video."
)

st.caption(
    "CropShield AI | Soybean classification research prototype | "
    "CPU inference powered by ONNX Runtime"
)
