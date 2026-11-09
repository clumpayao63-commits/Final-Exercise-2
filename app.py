import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Prediction System")
st.caption("A classroom prototype for preliminary red-wine quality classification.")

model_path = Path("wine_model.pkl")
data_path = Path("wine_quality.csv")

if not model_path.exists() or not data_path.exists():
    st.error("Required files are missing. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = pd.read_csv(data_path)

# Dataset information
c1, c2, c3 = st.columns(3)
c1.metric("Dataset records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write(
    "Upload a CSV file containing the 11 measurement columns. "
    "Do not include the original quality column."
)

required_columns = list(model.feature_names_in_)
template = data[required_columns].head(5)

st.download_button(
    label="Download sample input CSV",
    data=template.to_csv(index=False),
    file_name="sample_wine_input.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [column for column in required_columns if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            # Use the required columns in the same order as model training.
            input_data = new_data[required_columns].apply(
                pd.to_numeric, errors="coerce"
            )

            if input_data.isnull().all(axis=1).any():
                st.error("One or more rows contain no usable numeric measurements.")
            else:
                predictions = model.predict(input_data)
                results = new_data.copy()
                results["Prediction"] = [
                    "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                    for value in predictions
                ]

                st.subheader("Prediction results")
                st.dataframe(results, use_container_width=True)

                good_count = int((predictions == 1).sum())
                st.metric("Predicted good-quality samples", good_count)

                st.download_button(
                    label="Download prediction results",
                    data=results.to_csv(index=False),
                    file_name="wine_quality_predictions.csv",
                    mime="text/csv"
                )
    except Exception as error:
        st.error(f"Could not read or process the uploaded CSV: {error}")

st.warning(
    "This is a classroom screening prototype. Predictions are not a substitute "
    "for laboratory testing, sensory evaluation, food-safety procedures, or "
    "professional quality-control decisions."
)
