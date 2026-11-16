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
        box-shadow: 0 16px 42px rgba(92, 58, 145, 0.13);
        background:
            radial-gradient(circle at 92% 18%, rgba(255,255,255,0.9) 0 3%, transparent 3.2%),
            radial-gradient(circle at 84% 80%, rgba(203,180,245,0.28), transparent 25%),
            linear-gradient(118deg, #f3edff 0%, #eee5ff 48%, #fff8fd 100%);
    }

    .hero h1 {
        font-size: clamp(2.15rem, 4vw, 3rem);
        line-height: 1.12;
        max-width: 760px;
    }

    .hero p {
        max-width: 680px;
        line-height: 1.65;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.15rem;
    }

    .hero-pills span {
        display: inline-block;
        padding: 0.42rem 0.75rem;
        border: 1px solid rgba(126, 88, 185, 0.18);
        border-radius: 999px;
        background: rgba(255,255,255,0.72);
        color: #634493;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .hero-foot {
        margin-top: 1.15rem;
        letter-spacing: 0.08em;
        font-size: 0.7rem;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 12px 34px rgba(65, 39, 110, 0.09);
    }

    [data-testid="stSlider"] [role="slider"]:focus {
        box-shadow: 0 0 0 0.22rem rgba(112, 65, 184, 0.18);
    }

    .prediction-card {
        box-shadow: 0 10px 26px rgba(75, 45, 130, 0.07);
    }

    .section-heading {
        padding-bottom: 0.2rem;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.28rem 0.58rem;
        border-radius: 6px;
        background: #f3edff;
    }

    @media (max-width: 700px) {
        .hero-pills {
            gap: 0.4rem;
        }
        .hero-pills span {
            font-size: 0.7rem;
        }
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

    .section-heading {
        margin: 0.8rem 0 1.1rem 0;
    }

    .section-heading h2 {
        color: #30205f !important;
        font-size: 1.75rem;
        letter-spacing: -0.035em;
        margin: 0.15rem 0 0.2rem 0;
    }

    .section-heading p {
        color: #756b8d !important;
        margin: 0;
    }

    .eyebrow {
        color: #875ac4;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.13em;
    }

    .lower-heading {
        margin-top: 2rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e9e1f5 !important;
        border-radius: 20px !important;
        background: linear-gradient(180deg, #ffffff 0%, #fdfbff 100%) !important;
        box-shadow: 0 8px 28px rgba(65, 39, 110, 0.055);
        padding: 0.4rem;
    }

    .prediction-card {
        text-align: center;
        padding: 1.45rem 1rem 1.35rem 1rem;
        border-radius: 17px;
        margin: 0.6rem 0 1.1rem 0;
        border: 1px solid #e9defa;
        background: linear-gradient(145deg, #f7f2ff, #ffffff);
    }

    .prediction-card.setosa {
        background: linear-gradient(145deg, #effcf5, #ffffff);
        border-color: #d5f2e1;
    }

    .prediction-card.versicolor {
        background: linear-gradient(145deg, #fff8e9, #ffffff);
        border-color: #f6e7c5;
    }

    .prediction-card.virginica {
        background: linear-gradient(145deg, #fff0f6, #ffffff);
        border-color: #f4d8e6;
    }

    .prediction-kicker {
        color: #8065a8;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.13em;
    }

    .prediction-flower {
        font-size: 3.2rem;
        line-height: 1.35;
        margin-top: 0.45rem;
    }

    .prediction-name {
        color: #38216f;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .prediction-description {
        color: #756b8d;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #38216f !important;
    }

    [data-testid="stSlider"] {
        padding: 0.55rem 0.75rem 0.2rem 0.75rem;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.5rem;
        }

        .hero h1 {
            font-size: 1.8rem;
        }

        .prediction-name {
            font-size: 1.7rem;
        }
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

    /* --------------------------------------------------
       DARK BLACK + PURPLE THEME
       -------------------------------------------------- */
    :root { color-scheme: dark; }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #09090f !important;
        color: #e7e5ee !important;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 9, 15, 0.94) !important;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p,
    label,
    .stCaption {
        color: #b8b5c5 !important;
    }

    h1, h2, h3, h4,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #f5f3fa !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.25rem 2.4rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        border: 1px solid #34264e;
        background:
            radial-gradient(circle at 88% 20%, rgba(167,139,250,.18), transparent 28%),
            radial-gradient(circle at 10% 110%, rgba(124,58,237,.16), transparent 34%),
            linear-gradient(125deg, #171321 0%, #101019 58%, #171126 100%);
        box-shadow: 0 18px 55px rgba(0,0,0,.32);
    }

    .hero::after {
        content: "✿";
        position: absolute;
        right: 3rem;
        top: .2rem;
        font-size: 9rem;
        color: rgba(167,139,250,.12);
        pointer-events: none;
    }

    .hero-tag {
        display: inline-block;
        padding: .4rem .8rem;
        margin-bottom: .9rem;
        border-radius: 999px;
        background: rgba(167,139,250,.12);
        border: 1px solid rgba(167,139,250,.38);
        color: #c4b5fd !important;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .1em;
    }

    .hero h1 {
        color: #f7f5ff !important;
        font-size: clamp(2.2rem, 4vw, 3.15rem);
        letter-spacing: -.045em;
        line-height: 1.12;
        margin: 0 0 .55rem;
    }

    .hero p {
        color: #c4c0d0 !important;
        max-width: 700px;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .hero-pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.2rem; }
    .hero-pills span {
        padding: .43rem .78rem;
        border: 1px solid #403354;
        border-radius: 999px;
        background: rgba(255,255,255,.045);
        color: #ddd6fe !important;
        font-size: .78rem;
        font-weight: 650;
    }
    .hero-foot { color:#9993a9 !important; margin-top:1.1rem; font-size:.75rem; letter-spacing:.08em; }

    .section-heading { margin: 1rem 0 1.15rem; }
    .section-heading h2 { color:#f5f3fa !important; font-size:1.8rem; letter-spacing:-.035em; }
    .section-heading p, .section-note { color:#a7a2b4 !important; }
    .eyebrow {
        color:#bda4ff !important;
        background:rgba(139,92,246,.12);
        border:1px solid rgba(167,139,250,.22);
        border-radius:6px;
        padding:.3rem .6rem;
        font-size:.69rem;
        font-weight:800;
        letter-spacing:.13em;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(155deg,#15141e 0%,#101018 100%) !important;
        border: 1px solid #30283e !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 32px rgba(0,0,0,.22) !important;
        padding: .45rem;
        transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color:#62469a !important;
        box-shadow:0 14px 36px rgba(0,0,0,.3) !important;
        transform:translateY(-1px);
    }
    [data-testid="stVerticalBlockBorderWrapper"] h3 { color:#eee9fa !important; }

    [data-testid="stSlider"] {
        background:#11111a !important;
        border:1px solid #292536 !important;
        border-radius:13px;
        padding:.8rem 1rem .35rem;
        margin-bottom:.55rem;
    }
    [data-testid="stSlider"] p { color:#d0cbd9 !important; font-weight:600; }
    [data-testid="stSlider"] [role="slider"] {
        background:#a78bfa !important;
        border-color:#a78bfa !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background:#8b5cf6 !important;
    }
    [data-testid="stSlider"] [role="slider"]:focus {
        box-shadow:0 0 0 .22rem rgba(167,139,250,.25) !important;
    }

    /* Keep every Streamlit button, including Clear history, on-theme. */
    [data-testid="stButton"] > button,
    [data-testid="stButton"] button,
    div.stButton > button,
    div.stButton > button[kind="secondary"],
    div.stButton > button[kind="tertiary"] {
        background:#211a2e !important;
        background-color:#211a2e !important;
        color:#e5dcff !important;
        border:1px solid #51406b !important;
        border-radius:12px !important;
        min-height:2.8rem;
        font-weight:700;
        box-shadow:none !important;
    }
    [data-testid="stButton"] > button:hover,
    [data-testid="stButton"] button:hover,
    div.stButton > button:hover {
        background:#302343 !important;
        background-color:#302343 !important;
        color:#ffffff !important;
        border-color:#a78bfa !important;
    }
    [data-testid="stButton"] > button:focus,
    [data-testid="stButton"] button:focus {
        box-shadow:0 0 0 .18rem rgba(167,139,250,.25) !important;
    }

    div.stButton > button {
        background:#211a2e !important;
        color:#e5dcff !important;
        border:1px solid #51406b !important;
        border-radius:12px !important;
        min-height:2.8rem;
        font-weight:700;
        box-shadow:none !important;
    }
    div.stButton > button:hover {
        background:#302343 !important;
        color:#ffffff !important;
        border-color:#a78bfa !important;
    }
    div.stButton > button:focus { box-shadow:0 0 0 .18rem rgba(167,139,250,.25) !important; }

    .prediction-card {
        text-align:center;
        padding:1.5rem 1rem 1.4rem;
        border-radius:17px;
        margin:.6rem 0 1.1rem;
        border:1px solid #57417b;
        background:linear-gradient(145deg,#21182f,#15131e 78%);
        box-shadow:0 12px 30px rgba(0,0,0,.2);
    }
    .prediction-card.setosa { background:linear-gradient(145deg,#14251f,#15131e 80%); border-color:#315b49; }
    .prediction-card.versicolor { background:linear-gradient(145deg,#292218,#15131e 80%); border-color:#6c5632; }
    .prediction-card.virginica { background:linear-gradient(145deg,#2a1929,#15131e 80%); border-color:#70415f; }
    .prediction-kicker { color:#c4b5fd !important; font-size:.68rem; font-weight:800; letter-spacing:.14em; }
    .prediction-flower { font-size:3.25rem; line-height:1.35; margin-top:.45rem; }
    .prediction-name { color:#e9ddff !important; font-size:2rem; font-weight:850; letter-spacing:-.04em; }
    .prediction-description { color:#b6afc4 !important; font-size:.9rem; margin-top:.3rem; }

    div[data-testid="stMetric"] {
        background:linear-gradient(145deg,#171521,#11111a) !important;
        border:1px solid #342b46 !important;
        border-radius:16px;
        padding:1rem 1.1rem;
        box-shadow:0 6px 20px rgba(0,0,0,.2);
    }
    div[data-testid="stMetricLabel"] p { color:#aaa3b8 !important; }
    div[data-testid="stMetricValue"] { color:#d8c6ff !important; }

    [data-testid="stTabs"] button[role="tab"] { color:#aaa4b6 !important; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color:#c4b5fd !important;
        border-bottom-color:#a78bfa !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background:#11111a !important;
        border:1px solid #30283e !important;
        border-radius:12px;
    }
    [data-testid="stDataFrame"] * { color:#e5e1ed !important; }
    [data-testid="stExpander"] {
        background:#14131d !important;
        border:1px solid #342b46 !important;
        border-radius:14px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p { color:#e5ddf5 !important; }
    [data-testid="stAlert"] { background:#191626 !important; border:1px solid #44345f !important; border-radius:13px; }
    [data-testid="stAlert"] p { color:#d7d1e2 !important; }

    input, textarea, [data-baseweb="select"] > div {
        background:#171621 !important;
        color:#f0edf7 !important;
        border-color:#40364f !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"] { background:#1a1824 !important; }
    [data-baseweb="menu"] * { color:#eee9f7 !important; }

    hr { border-color:#30283e !important; }
    .footer { color:#858092 !important; text-align:center; padding:1.6rem 0 .5rem; font-size:.84rem; }

    @media (max-width:700px) {
        .hero { padding:1.5rem; }
        .hero h1 { font-size:1.85rem; }
        .hero::after { right:1rem; font-size:5.5rem; }
        .prediction-name { font-size:1.7rem; }
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
    <div class="hero-tag">✦ MACHINE LEARNING · INTERACTIVE DEMO</div>
    <h1>🌸 Iris Flower Classifier</h1>
    <p>Every flower has a story. Explore its measurements and let machine learning reveal its species.</p>
    <div class="hero-pills">
        <span>✿ 150 flower samples</span>
        <span>✿ 3 Iris species</span>
        <span>✿ 4 measurements</span>
    </div>
    <div class="hero-foot">BUILT WITH PYTHON &nbsp;·&nbsp; SCIKIT-LEARN &nbsp;·&nbsp; STREAMLIT</div>
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


# 7. INTERACTIVE CLASSIFIER WORKSPACE
# --------------------------------------------------

st.markdown("""
<div class="section-heading">
    <span class="eyebrow">FLOWER IDENTIFICATION</span>
    <h2>Explore your flower</h2>
    <p>Choose a preset or fine-tune the measurements. Your result updates live.</p>
</div>
""", unsafe_allow_html=True)

input_panel, result_panel = st.columns([1.05, 0.95], gap="large")

with input_panel:
    with st.container(border=True):
        st.markdown("### 🎛️ Flower measurements")
        st.caption("All measurements are in centimetres.")

        st.markdown("**Quick examples**")
        sample1, sample2, sample3 = st.columns(3)

        with sample1:
            st.button(
                "🌱 Setosa",
                use_container_width=True,
                on_click=load_sample,
                args=(5.1, 3.5, 1.4, 0.2)
            )

        with sample2:
            st.button(
                "🌼 Versicolor",
                use_container_width=True,
                on_click=load_sample,
                args=(6.0, 2.9, 4.5, 1.5)
            )

        with sample3:
            st.button(
                "🌺 Virginica",
                use_container_width=True,
                on_click=load_sample,
                args=(6.5, 3.0, 5.5, 2.0)
            )

        st.markdown("---")
        st.markdown("**Sepal dimensions**")

        sepal_col1, sepal_col2 = st.columns(2)

        with sepal_col1:
            st.slider(
                "Sepal length",
                min_value=0.0,
                max_value=8.0,
                step=0.1,
                key="sepal_length"
            )

        with sepal_col2:
            st.slider(
                "Sepal width",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                key="sepal_width"
            )

        st.markdown("**Petal dimensions**")

        petal_col1, petal_col2 = st.columns(2)

        with petal_col1:
            st.slider(
                "Petal length",
                min_value=0.0,
                max_value=7.0,
                step=0.1,
                key="petal_length"
            )

        with petal_col2:
            st.slider(
                "Petal width",
                min_value=0.0,
                max_value=3.0,
                step=0.1,
                key="petal_width"
            )

        st.caption("Tip: Try the sample buttons first, then move a slider to see how the prediction changes.")


# --------------------------------------------------
# 8. LIVE MODEL PREDICTION
# --------------------------------------------------

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

vote_share = max(probability_by_species.values())

current_measurements = {
    "Sepal length": st.session_state.sepal_length,
    "Sepal width": st.session_state.sepal_width,
    "Petal length": st.session_state.petal_length,
    "Petal width": st.session_state.petal_width
}

previous_prediction = st.session_state.latest_prediction

if (
    previous_prediction is None
    or previous_prediction["measurements"] != current_measurements
):
    prediction_record = {
        "time": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        "species": str(predicted_species),
        "vote_share": vote_share,
        "probabilities": probability_by_species,
        "measurements": current_measurements
    }

    st.session_state.latest_prediction = prediction_record
    st.session_state.prediction_history.insert(0, {
        "Time": prediction_record["time"],
        "Predicted species": str(predicted_species).replace("Iris-", "").title(),
        "Top neighbour vote": f"{vote_share:.0f}%"
    })


# --------------------------------------------------
# 9. LIVE RESULT PANEL
# --------------------------------------------------

with result_panel:
    result = st.session_state.latest_prediction
    display_species = result["species"].replace("Iris-", "").title()

    with st.container(border=True):
        st.markdown("### ✨ Your prediction")
        st.caption("Live result · Updates with every measurement change")

        species_details = {
            "Setosa": {
                "emoji": "🌱",
                "description": "Recognized by its notably short petals.",
                "class_name": "setosa"
            },
            "Versicolor": {
                "emoji": "🌼",
                "description": "A middle-sized flower with balanced petal dimensions.",
                "class_name": "versicolor"
            },
            "Virginica": {
                "emoji": "🌺",
                "description": "Typically has the largest petal dimensions.",
                "class_name": "virginica"
            }
        }

        details = species_details.get(display_species, {
            "emoji": "🌸",
            "description": "The model has identified this Iris species.",
            "class_name": "other"
        })

        st.markdown(
            f"""
            <div class="prediction-card {details['class_name']}">
                <div class="prediction-kicker">PREDICTED SPECIES</div>
                <div class="prediction-flower">{details['emoji']}</div>
                <div class="prediction-name">{display_species}</div>
                <div class="prediction-description">{details['description']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Nearest-neighbour voting**")
        st.metric("Top species vote", f"{result['vote_share']:.0f}%")

        probability_df = pd.DataFrame({
            "Species": [
                name.replace("Iris-", "").title()
                for name in result["probabilities"].keys()
            ],
            "Neighbour vote (%)": list(result["probabilities"].values())
        }).set_index("Species")

        st.bar_chart(probability_df, y="Neighbour vote (%)", height=190)

        st.caption(
            "The percentage is the share of the 5 nearest training neighbours "
            "voting for this class. It is not a calibrated probability."
        )


# --------------------------------------------------
# MODEL OVERVIEW
# --------------------------------------------------

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Test accuracy", f"{accuracy * 100:.2f}%")
metric2.metric("Flowers in dataset", f"{len(df):,}")
metric3.metric("Species", f"{df['Species'].nunique()}")

st.divider()


st.markdown("""
<div class="section-heading lower-heading">
    <span class="eyebrow">MODEL INSIGHTS</span>
    <h2>Understand the dataset</h2>
    <p>Compare the three species and review your recent predictions.</p>
</div>
""", unsafe_allow_html=True)


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
