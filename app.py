import streamlit as st
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)


# --------------------------------------------------
# 2. LIGHT THEME STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f6f8fc;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 252, 0.9);
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(115deg, #e0f2fe 0%, #ede9fe 55%, #fce7f3 100%);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.3rem;
    }

    .hero h1 {
        color: #172554;
        font-size: 2.35rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.04em;
    }

    .hero p {
        color: #475569;
        font-size: 1.05rem;
        margin: 0;
    }

    .section-note {
        color: #64748b;
        margin-top: -0.45rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
    }

    div[data-testid="stMetricLabel"] p {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #172554;
    }

    div.stButton > button {
        border-radius: 11px;
        min-height: 2.7rem;
        font-weight: 600;
        border: 1px solid #dbe3ef;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: 1px solid #2563eb;
        color: white;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 16px;
        border-color: #e5eaf2;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

default_measurements = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

for key, value in default_measurements.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_sample(sepal_length, sepal_width, petal_length, petal_width):
    """Load a sample flower's measurements into the sliders."""
    st.session_state.sepal_length = sepal_length
    st.session_state.sepal_width = sepal_width
    st.session_state.petal_length = petal_length
    st.session_state.petal_width = petal_width


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🌸 Iris Flower Classifier</h1>
    <p>Explore flower measurements and predict an Iris species with Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 5. LOAD DATA AND TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def train_model():
    df = pd.read_csv("Iris.csv")

    feature_columns = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]

    X = df[feature_columns]
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    test_predictions = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return model, scaler, test_accuracy, df, feature_columns


try:
    model, scaler, accuracy, df, feature_columns = train_model()
except Exception as error:
    st.error(f"Could not load the Iris dataset or train the model: {error}")
    st.info("Check that Iris.csv is present in the root of this GitHub repository.")
    st.stop()


# --------------------------------------------------
# 6. MODEL OVERVIEW
# --------------------------------------------------

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Test accuracy", f"{accuracy * 100:.2f}%")
metric2.metric("Flowers in dataset", f"{len(df):,}")
metric3.metric("Species", f"{df['Species'].nunique()}")


st.divider()


# --------------------------------------------------
# 7. SAMPLE FLOWER BUTTONS
# --------------------------------------------------

st.subheader("1. Choose a sample flower")
st.markdown(
    '<p class="section-note">Select an example to fill the measurement sliders automatically.</p>',
    unsafe_allow_html=True
)

sample1, sample2, sample3 = st.columns(3)

with sample1:
    st.button(
        "🌱 Setosa example",
        use_container_width=True,
        on_click=load_sample,
        args=(5.1, 3.5, 1.4, 0.2)
    )

with sample2:
    st.button(
        "🌼 Versicolor example",
        use_container_width=True,
        on_click=load_sample,
        args=(6.0, 2.9, 4.5, 1.5)
    )

with sample3:
    st.button(
        "🌺 Virginica example",
        use_container_width=True,
        on_click=load_sample,
        args=(6.5, 3.0, 5.5, 2.0)
    )


# --------------------------------------------------
# 8. INTERACTIVE MEASUREMENTS
# --------------------------------------------------

st.subheader("2. Adjust flower measurements")
st.markdown(
    '<p class="section-note">Move the sliders to enter the flower dimensions in centimetres.</p>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

with left:
    st.slider(
        "Sepal length (cm)",
        min_value=0.0,
        max_value=8.0,
        step=0.1,
        key="sepal_length"
    )

    st.slider(
        "Sepal width (cm)",
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        key="sepal_width"
    )

with right:
    st.slider(
        "Petal length (cm)",
        min_value=0.0,
        max_value=7.0,
        step=0.1,
        key="petal_length"
    )

    st.slider(
        "Petal width (cm)",
        min_value=0.0,
        max_value=3.0,
        step=0.1,
        key="petal_width"
    )


# --------------------------------------------------
# 9. PREDICTION
# --------------------------------------------------

st.subheader("3. Predict the species")

if st.button("🔍 Predict flower species", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[
            st.session_state.sepal_length,
            st.session_state.sepal_width,
            st.session_state.petal_length,
            st.session_state.petal_width
        ]],
        columns=feature_columns
    )

    input_scaled = scaler.transform(input_data)
    predicted_species = model.predict(input_scaled)[0]
    class_probabilities = model.predict_proba(input_scaled)[0]

    probability_by_species = {
        str(species): float(probability * 100)
        for species, probability in zip(model.classes_, class_probabilities)
    }

    # For KNN, this percentage represents the share of the five
    # nearest neighbours voting for the predicted class.
    vote_share = max(probability_by_species.values())

    prediction_record = {
        "time": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        "species": str(predicted_species),
        "vote_share": vote_share,
        "probabilities": probability_by_species,
        "measurements": {
            "Sepal length": st.session_state.sepal_length,
            "Sepal width": st.session_state.sepal_width,
            "Petal length": st.session_state.petal_length,
            "Petal width": st.session_state.petal_width
        }
    }

    st.session_state.latest_prediction = prediction_record
    st.session_state.prediction_history.insert(0, {
        "Time": prediction_record["time"],
        "Predicted species": str(predicted_species).replace("Iris-", "").title(),
        "Top neighbour vote": f"{vote_share:.0f}%"
    })


# --------------------------------------------------
# 10. DISPLAY PREDICTION RESULT
# --------------------------------------------------

if st.session_state.latest_prediction is not None:
    result = st.session_state.latest_prediction
    display_species = result["species"].replace("Iris-", "").title()

    st.markdown("### Prediction result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.success(f"Predicted species: **{display_species}**")

    with result_col2:
        st.metric("Top neighbour vote", f"{result['vote_share']:.0f}%")

    st.caption(
        "For this KNN model, the percentage is the share of the 5 nearest "
        "training neighbours voting for the predicted species. It is not a "
        "calibrated probability or a guarantee of correctness."
    )

    probability_df = pd.DataFrame({
        "Species": [
            name.replace("Iris-", "").title()
            for name in result["probabilities"].keys()
        ],
        "Neighbour vote (%)": list(result["probabilities"].values())
    }).set_index("Species")

    st.markdown("#### Votes across the three species")
    st.bar_chart(probability_df, y="Neighbour vote (%)")


st.divider()


# --------------------------------------------------
# 11. SPECIES COMPARISON
# --------------------------------------------------

st.subheader("🌷 Compare Iris species")
st.markdown(
    '<p class="section-note">Average measurements calculated from the dataset.</p>',
    unsafe_allow_html=True
)

comparison = (
    df.groupby("Species")[feature_columns]
    .mean()
    .round(2)
)

comparison.index = [
    str(species).replace("Iris-", "").title()
    for species in comparison.index
]

tab1, tab2 = st.tabs(["Measurement table", "Petal comparison"])

with tab1:
    st.dataframe(comparison, use_container_width=True)

with tab2:
    st.bar_chart(comparison[["PetalLengthCm", "PetalWidthCm"]])


# --------------------------------------------------
# 12. PREDICTION HISTORY
# --------------------------------------------------

st.divider()

history_heading, history_action = st.columns([3, 1])

with history_heading:
    st.subheader("📋 Prediction history")

with history_action:
    if st.button("Clear history", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    st.caption("History is kept for the current browser session and is not saved permanently.")
else:
    st.info("Your predictions will appear here after you make them.")


# --------------------------------------------------
# 13. ABOUT THE PROJECT
# --------------------------------------------------

with st.expander("About this Machine Learning project"):
    st.write("""
    **Goal:** Predict the species of an Iris flower from four measurements.

    **Algorithm:** K-Nearest Neighbors (KNN), with 5 neighbours.

    **Preprocessing:** StandardScaler is fitted on the training data and
    then applied to the test data and user inputs.

    **Evaluation:** The model is evaluated on a stratified 20% test split.

    **Dataset:** Iris.csv, containing sepal and petal measurements for
    Setosa, Versicolor and Virginica flowers.
    """)


# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with Python, Pandas, Scikit-learn and Streamlit
    <br>
    Iris Flower Classification | Machine Learning Portfolio Project
</div>
""", unsafe_allow_html=True)
