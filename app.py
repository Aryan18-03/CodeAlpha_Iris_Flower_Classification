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
    :root {
        color-scheme: light;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #ffffff !important;
        color: #24213a !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96) !important;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p,
    label {
        color: #49445f !important;
    }

    h1, h2, h3, h4,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #30205f !important;
    }

    .hero {
        padding: 2.1rem 2.3rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #f5f0ff 0%, #eee7ff 55%, #faf5ff 100%);
        border: 1px solid #e6dcff;
        margin-bottom: 1.3rem;
    }

    .hero h1 {
        color: #38216f !important;
        font-size: 2.35rem;
        margin: 0 0 0.45rem 0;
        letter-spacing: -0.04em;
    }

    .hero p {
        color: #62557d !important;
        font-size: 1.05rem;
        margin: 0;
    }

    .hero {
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 32px rgba(92, 58, 145, 0.08);
    }

    .hero::after {
        content: "✿";
        position: absolute;
        right: 3.2rem;
        top: 0.35rem;
        font-size: 8rem;
        line-height: 1;
        color: rgba(112, 65, 184, 0.09);
        pointer-events: none;
    }

    .hero-tag {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin-bottom: 0.85rem;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid #e4d8fb;
        color: #7041b8;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.09em;
    }

    .hero-foot {
        margin-top: 1.05rem;
        color: #786a96;
        font-size: 0.83rem;
        font-weight: 600;
    }

    .section-note {
        color: #756b8d !important;
        margin-top: -0.45rem;
    }

    [data-testid="stSubheader"] {
        padding-top: 0.35rem;
        padding-bottom: 0.15rem;
    }

    [data-testid="stSlider"] {
        background: #fbf9ff;
        border: 1px solid #eee7fa;
        border-radius: 13px;
        padding: 0.8rem 1rem 0.35rem 1rem;
        margin-bottom: 0.55rem;
    }

    [data-testid="stSlider"] p {
        color: #49336f !important;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(75, 45, 130, 0.10);
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    [data-testid="stDataFrame"] {
        box-shadow: 0 3px 12px rgba(75, 45, 130, 0.04);
    }

    hr {
        border-color: #eee7f7 !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e8e1f5 !important;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 16px rgba(75, 45, 130, 0.06);
    }

    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"] {
        color: #38216f !important;
    }

    div.stButton > button,
    div.stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #6336a5 !important;
        border: 1px solid #d9c9f5 !important;
        border-radius: 12px !important;
        min-height: 2.8rem;
        font-weight: 650;
        box-shadow: 0 2px 8px rgba(75, 45, 130, 0.05);
    }

    div.stButton > button:hover {
        background: #f5efff !important;
        color: #4b2488 !important;
        border-color: #a98add !important;
    }

    div.stButton > button[kind="primary"] {
        background: #7041b8 !important;
        color: #ffffff !important;
        border: 1px solid #7041b8 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #5c329e !important;
        color: #ffffff !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: #7041b8 !important;
        border-color: #7041b8 !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: #7041b8 !important;
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: #62557d !important;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #6336a5 !important;
        border-bottom-color: #7041b8 !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #e8e1f5;
        border-radius: 12px;
    }

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e8e1f5 !important;
        border-radius: 14px !important;
    }

    .footer {
        text-align: center;
        color: #9388aa !important;
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
    <div class="hero-tag">MACHINE LEARNING · INTERACTIVE DEMO</div>
    <h1>🌸 Iris Flower Classifier</h1>
    <p>Discover the species behind each flower through four simple measurements.</p>
    <div class="hero-foot">Powered by Python · Scikit-learn · K-Nearest Neighbors</div>
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
