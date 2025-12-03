from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image
import streamlit as st


# Ensure the src directory is on sys.path so we can import the package
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr_project.io.dataset_loader import EmnistLoader  # noqa: E402
from ocr_project.preprocess.transformer import Transformer  # noqa: E402
from ocr_project.utils import (  # noqa: E402
    emnist_label_to_letter,
    prepare_single_image_for_model,
)
from ocr_project import config  # noqa: E402


@st.cache_resource
def load_test_data():
    loader = EmnistLoader(config.EMNIST_LETTERS_TRAIN, config.EMNIST_LETTERS_TEST)
    test_split = loader.load_test()
    images = test_split.images
    labels = test_split.labels
    X_test = Transformer.normalize(Transformer.flatten(images))
    return images, labels, X_test


@st.cache_resource
def load_models():
    dt_artifact = joblib.load(PROJECT_ROOT / "artifacts" / "decision_tree.pkl")
    rf_artifact = joblib.load(PROJECT_ROOT / "artifacts" / "random_forest.pkl")

    dt_model = dt_artifact["model"]
    rf_model = rf_artifact["model"]
    dt_acc = dt_artifact.get("accuracy")
    rf_acc = rf_artifact.get("accuracy")
    return dt_model, rf_model, dt_acc, rf_acc


def main() -> None:
    st.title("Handwritten Letter OCR – Decision Tree vs Random Forest")

    images, labels, X_test = load_test_data()
    dt_model, rf_model, dt_acc, rf_acc = load_models()

    # Sidebar: aggregate comparison
    st.sidebar.header("Model comparison")
    if dt_acc is not None and rf_acc is not None:
        st.sidebar.write(f"Decision Tree accuracy: **{dt_acc:.3%}**")
        st.sidebar.write(f"Random Forest accuracy: **{rf_acc:.3%}**")
        diff = rf_acc - dt_acc
        pct_diff = diff / dt_acc * 100 if dt_acc != 0 else 0.0
        st.sidebar.write(f"Absolute difference (RF - DT): **{diff:.3%}**")
        st.sidebar.write(f"Relative improvement of RF over DT: **{pct_diff:.2f}%**")
    else:
        st.sidebar.write("Train and save models first (run scripts/train_two_models.py).")

    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Mode",
        ["Browse test set", "Upload your own image"],
        index=0,
    )

    if mode == "Browse test set":
        st.header("Browse EMNIST test images")

        idx = st.slider("Test sample index", 0, len(images) - 1, 0)
        img = images[idx]
        true_label = int(labels[idx])
        true_letter = emnist_label_to_letter(true_label)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"Test image #{idx}", width=150)
        with col2:
            st.write(f"True label: **{true_label}** → **'{true_letter}'**")

        x = X_test[idx : idx + 1]
        dt_pred = int(dt_model.predict(x)[0])
        rf_pred = int(rf_model.predict(x)[0])
        dt_letter = emnist_label_to_letter(dt_pred)
        rf_letter = emnist_label_to_letter(rf_pred)

        st.subheader("Model predictions")
        st.write(f"Decision Tree: **{dt_pred}** → **'{dt_letter}'**")
        st.write(f"Random Forest: **{rf_pred}** → **'{rf_letter}'**")

        correct_dt = dt_pred == true_label
        correct_rf = rf_pred == true_label
        st.write(
            f"Decision Tree correct? **{correct_dt}** | "
            f"Random Forest correct? **{correct_rf}**"
        )
    else:
        st.header("Upload your own 28×28 grayscale letter")

        uploaded_file = st.file_uploader(
            "Upload an image (will be converted to 28×28 grayscale)",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert("L")
            img_pil = img_pil.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

            st.image(img_pil, caption="Uploaded image (resized to 28×28)", width=150)

            img_arr = np.array(img_pil)
            x = prepare_single_image_for_model(img_arr)

            dt_pred = int(dt_model.predict(x)[0])
            rf_pred = int(rf_model.predict(x)[0])
            dt_letter = emnist_label_to_letter(dt_pred)
            rf_letter = emnist_label_to_letter(rf_pred)

            st.subheader("Model predictions on your image")
            st.write(f"Decision Tree: **{dt_pred}** → **'{dt_letter}'**")
            st.write(f"Random Forest: **{rf_pred}** → **'{rf_letter}'**")

            if dt_pred == rf_pred:
                st.success("Both models agree on the predicted letter.")
            else:
                st.info(
                    "The models disagree on this example – this may be interesting for error analysis."
                )


if __name__ == "__main__":
    main()


