from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


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


FEATURE_DIRS = {
    "pixels": PROJECT_ROOT / "artifacts_pixels",
    "hog": PROJECT_ROOT / "artifacts_hog",
    "pca": PROJECT_ROOT / "artifacts_pca",
    # Default "strong" pipeline: combined HOG + PCA features.
    # Artifacts are saved by OCRPipeline.run_default_pipeline() into `artifacts/`.
    "hog_pca": PROJECT_ROOT / "artifacts",
}


@st.cache_resource
def load_models(feature_type: str):
    """
    Load models + feature extractor for a given feature type.
    """
    base_dir = FEATURE_DIRS[feature_type]
    dt_artifact = joblib.load(base_dir / "decision_tree.pkl")
    rf_artifact = joblib.load(base_dir / "random_forest.pkl")

    dt_model = dt_artifact["model"]
    rf_model = rf_artifact["model"]
    dt_acc = dt_artifact.get("accuracy")
    rf_acc = rf_artifact.get("accuracy")

    metadata_dt = dt_artifact.get("metadata", {}) or {}
    metadata_rf = rf_artifact.get("metadata", {}) or {}
    feature_extractor = metadata_dt.get("feature_extractor") or metadata_rf.get(
        "feature_extractor"
    )

    return dt_model, rf_model, dt_acc, rf_acc, feature_extractor


@st.cache_resource
def get_transformed_test_features(feature_type: str):
    """
    Return test images, labels and **feature-transformed** test matrix for a given
    feature type.

    This runs the expensive `feature_extractor.transform` over the full test set
    only once per feature type instead of on every Streamlit rerun (e.g. when
    you move the sample slider).
    """
    images, labels, X_test_raw = load_test_data()
    _, _, _, _, feature_extractor = load_models(feature_type)

    if feature_extractor is not None:
        X_test = feature_extractor.transform(X_test_raw)
    else:
        X_test = X_test_raw

    return images, labels, X_test


def get_top_predictions(proba: np.ndarray, model_classes: np.ndarray, top_k: int = 10) -> list[tuple[int, str, float]]:
    """
    Get top K predictions with their labels, letters, and confidence percentages.
    
    Args:
        proba: Probability array from predict_proba() (ordered by model.classes_)
        model_classes: The classes_ attribute from the model (maps proba index to label)
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples: (label, letter, confidence_percentage)
    """
    # Get indices sorted by probability (descending)
    top_indices = np.argsort(proba)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        # Map probability index to actual class label
        label = int(model_classes[idx])
        letter = emnist_label_to_letter(label)
        confidence = proba[idx] * 100
        results.append((label, letter, confidence))
    
    return results


def load_all_accuracies() -> dict[str, dict[str, float | None]]:
    """
    Read accuracies for all (feature_type, model) combos from artifacts.
    """
    results: dict[str, dict[str, float | None]] = {}
    for ft, base_dir in FEATURE_DIRS.items():
        try:
            dt_artifact = joblib.load(base_dir / "decision_tree.pkl")
            rf_artifact = joblib.load(base_dir / "random_forest.pkl")
        except FileNotFoundError:
            continue

        results[ft] = {
            "decision_tree": dt_artifact.get("accuracy"),
            "random_forest": rf_artifact.get("accuracy"),
        }
    return results


def main() -> None:
    st.title("Handwritten Letter OCR – Decision Tree vs Random Forest")

    # Cache-heavy test features (including any HOG/PCA transforms) per feature type
    # so slider changes don't recompute them every time.

    # Sidebar: high-level navigation + aggregate metrics
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Interactive demo", "Evaluation & graphs"],
        index=0,
    )

    st.sidebar.markdown("---")
    feature_labels = {
        "pixels": "Pixels only",
        "hog": "HOG features",
        "pca": "PCA features",
        "hog_pca": "HOG + PCA (combined)",
    }
    feature_type = st.sidebar.selectbox(
        "Feature type",
        options=list(FEATURE_DIRS.keys()),
        format_func=lambda k: feature_labels.get(k, k),
    )

    dt_model, rf_model, dt_acc, rf_acc, feature_extractor = load_models(feature_type)
    images, labels, X_test = get_transformed_test_features(feature_type)

    # Sidebar: aggregate comparison
    st.sidebar.header("Model comparison (current feature type)")
    if dt_acc is not None and rf_acc is not None:
        st.sidebar.write(f"Decision Tree accuracy: **{dt_acc:.3%}**")
        st.sidebar.write(f"Random Forest accuracy: **{rf_acc:.3%}**")
        diff = rf_acc - dt_acc
        pct_diff = diff / dt_acc * 100 if dt_acc != 0 else 0.0
        st.sidebar.write(f"Absolute difference (RF - DT): **{diff:.3%}**")
        st.sidebar.write(f"Relative improvement of RF over DT: **{pct_diff:.2f}%**")
    else:
        st.sidebar.write("Train and save models first via the notebook.")

    if page == "Interactive demo":
        st.sidebar.markdown("---")
        mode = st.sidebar.radio(
            "Mode",
            ["Browse test set", "Upload your own image"],
            index=0,
        )
    else:
        mode = None

    if page == "Interactive demo" and mode == "Browse test set":
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
        
        # Get confidence scores
        dt_proba = dt_model.predict_proba(x)[0]
        rf_proba = rf_model.predict_proba(x)[0]
        
        # Map predictions to probability indices using classes_
        dt_pred_idx = np.where(dt_model.classes_ == dt_pred)[0][0]
        rf_pred_idx = np.where(rf_model.classes_ == rf_pred)[0][0]
        dt_confidence = dt_proba[dt_pred_idx] * 100
        rf_confidence = rf_proba[rf_pred_idx] * 100

        st.subheader("Model predictions")

        correct_dt = dt_pred == true_label
        correct_rf = rf_pred == true_label

        if correct_dt:
            st.success(
                f"Decision Tree: **{dt_pred}** → **'{dt_letter}'** "
                f"(confidence: **{dt_confidence:.2f}%**) ✅ (correct)"
            )
        else:
            st.error(
                f"Decision Tree: **{dt_pred}** → **'{dt_letter}'** "
                f"(confidence: **{dt_confidence:.2f}%**) ❌ "
                f"(true label: **{true_label}** → **'{true_letter}'**)"
            )

        if correct_rf:
            st.success(
                f"Random Forest: **{rf_pred}** → **'{rf_letter}'** "
                f"(confidence: **{rf_confidence:.2f}%**) ✅ (correct)"
            )
        else:
            st.error(
                f"Random Forest: **{rf_pred}** → **'{rf_letter}'** "
                f"(confidence: **{rf_confidence:.2f}%**) ❌ "
                f"(true label: **{true_label}** → **'{true_letter}'**)"
            )
        
        # Top 10 predictions details
        st.markdown("---")
        with st.expander("📊 View Details: Top 10 Predictions", expanded=False):
            dt_top10 = get_top_predictions(dt_proba, dt_model.classes_, top_k=10)
            rf_top10 = get_top_predictions(rf_proba, rf_model.classes_, top_k=10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Decision Tree - Top 10 Guesses")
                for rank, (label, letter, conf) in enumerate(dt_top10, 1):
                    is_correct = label == true_label
                    marker = "✅" if is_correct else ""
                    st.write(
                        f"{rank}. Label **{label}** → **'{letter}'** "
                        f": **{conf:.2f}%** {marker}"
                    )
            
            with col2:
                st.subheader("Random Forest - Top 10 Guesses")
                for rank, (label, letter, conf) in enumerate(rf_top10, 1):
                    is_correct = label == true_label
                    marker = "✅" if is_correct else ""
                    st.write(
                        f"{rank}. Label **{label}** → **'{letter}'** "
                        f": **{conf:.2f}%** {marker}"
                    )

    elif page == "Interactive demo" and mode == "Upload your own image":
        st.header("Upload your own handwritten letter")

        uploaded_file = st.file_uploader(
            "Upload an image (any size/colour – it will be converted to a 28×28 grayscale EMNIST-style input)",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert("L")
            img_pil = img_pil.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

            st.image(img_pil, caption="Uploaded image (resized to 28×28)", width=150)

            img_arr = np.array(img_pil)
            x_raw = prepare_single_image_for_model(img_arr)

            # Apply the same feature transformation used at training time.
            if feature_extractor is not None:
                x = feature_extractor.transform(x_raw)
            else:
                x = x_raw

            dt_pred = int(dt_model.predict(x)[0])
            rf_pred = int(rf_model.predict(x)[0])
            dt_letter = emnist_label_to_letter(dt_pred)
            rf_letter = emnist_label_to_letter(rf_pred)
            
            # Get confidence scores
            dt_proba = dt_model.predict_proba(x)[0]
            rf_proba = rf_model.predict_proba(x)[0]
            
            # Map predictions to probability indices using classes_
            dt_pred_idx = np.where(dt_model.classes_ == dt_pred)[0][0]
            rf_pred_idx = np.where(rf_model.classes_ == rf_pred)[0][0]
            dt_confidence = dt_proba[dt_pred_idx] * 100
            rf_confidence = rf_proba[rf_pred_idx] * 100

            st.subheader("Model predictions on your image")
            st.info("No ground-truth label here, so we just show what each model predicts.")
            st.write(
                f"Decision Tree: **{dt_pred}** → **'{dt_letter}'** "
                f"(confidence: **{dt_confidence:.2f}%**)"
            )
            st.write(
                f"Random Forest: **{rf_pred}** → **'{rf_letter}'** "
                f"(confidence: **{rf_confidence:.2f}%**)"
            )

            if dt_pred == rf_pred:
                st.success("Both models agree on the predicted letter.")
            else:
                st.info(
                    "The models disagree on this example – this may be interesting for error analysis."
                )
            
            # Top 10 predictions details
            st.markdown("---")
            with st.expander("📊 View Details: Top 10 Predictions", expanded=False):
                dt_top10 = get_top_predictions(dt_proba, dt_model.classes_, top_k=10)
                rf_top10 = get_top_predictions(rf_proba, rf_model.classes_, top_k=10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Decision Tree - Top 10 Guesses")
                    for rank, (label, letter, conf) in enumerate(dt_top10, 1):
                        st.write(
                            f"{rank}. Label **{label}** → **'{letter}'** "
                            f": **{conf:.2f}%**"
                        )
                
                with col2:
                    st.subheader("Random Forest - Top 10 Guesses")
                    for rank, (label, letter, conf) in enumerate(rf_top10, 1):
                        st.write(
                            f"{rank}. Label **{label}** → **'{letter}'** "
                            f": **{conf:.2f}%**"
                        )

    elif page == "Evaluation & graphs":
        st.header("Evaluation & graphs")

        # --- Per-feature-type accuracy overview ---
        all_accs = load_all_accuracies()
        if all_accs:
            st.subheader("Accuracy overview (all feature types)")
            rows = []
            for ft, models in all_accs.items():
                for model_name, acc in models.items():
                    if acc is None:
                        continue
                    rows.append((feature_labels.get(ft, ft), model_name, acc))

            if rows:
                import pandas as pd

                df = pd.DataFrame(rows, columns=["Feature type", "Model", "Accuracy"])
                st.dataframe(
                    df.style.format({"Accuracy": "{:.3%}"}),
                    use_container_width=True,
                )

        # --- Detailed plots for the currently selected feature type ---
        st.subheader(
            f"Detailed evaluation for {feature_labels.get(feature_type, feature_type)}"
        )

        # Compute predictions on the full test set
        dt_preds = dt_model.predict(X_test)
        rf_preds = rf_model.predict(X_test)

        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            dt_acc_full = (dt_preds == labels).mean()
            st.metric("Decision Tree accuracy", f"{dt_acc_full:.3%}")
        with col2:
            rf_acc_full = (rf_preds == labels).mean()
            st.metric("Random Forest accuracy", f"{rf_acc_full:.3%}")

        # Confusion matrices
        st.subheader("Confusion matrices")
        cm_dt = confusion_matrix(labels, dt_preds)
        cm_rf = confusion_matrix(labels, rf_preds)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_dt, ax=ax1, cmap="Greens", annot=False)
        ax1.set_title("Decision Tree – Confusion Matrix")
        ax1.set_xlabel("Predicted label")
        ax1.set_ylabel("True label")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_rf, ax=ax2, cmap="Blues", annot=False)
        ax2.set_title("Random Forest – Confusion Matrix")
        ax2.set_xlabel("Predicted label")
        ax2.set_ylabel("True label")
        st.pyplot(fig2)

        # Classification reports
        st.subheader("Classification reports")
        st.markdown("**Decision Tree**")
        st.text(classification_report(labels, dt_preds, zero_division=0))

        st.markdown("**Random Forest**")
        st.text(classification_report(labels, rf_preds, zero_division=0))


if __name__ == "__main__":
    main()


