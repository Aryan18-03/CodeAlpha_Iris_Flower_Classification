
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Application title
st.title("🌸 Iris Flower Classification")
st.write(
    "Enter the flower measurements below "
    "to predict its species using Machine Learning."
)

# Load dataset
@st.cache_resource
def train_model():
    df = pd.read_csv("Iris.csv")

    X = df.drop(["Id", "Species"], axis=1)
    y = df["Species"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    return model, scaler, accuracy


# Train model
try:
    model, scaler, accuracy = train_model()

    st.success("Machine Learning model loaded successfully!")

    st.metric("Model Test Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Enter Flower Measurements")

    # User input
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.0,
        max_value=10.0, value=5.1, step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.0,
        max_value=10.0, value=3.5, step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.0,
        max_value=10.0, value=1.4, step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.0,
        max_value=10.0, value=0.2, step=0.1
    )

    # Prediction button
    if st.button("Predict Species", type="primary"):

        input_data = pd.DataFrame(
            [[sepal_length, sepal_width,
              petal_length, petal_width]],
            columns=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )

        # Apply the same scaling used during training
        input_scaled = scaler.transform(input_data)

        # Predict species
        prediction = model.predict(input_scaled)

        st.subheader("Prediction Result")
        st.success(f"Predicted Species: {prediction[0]}")

except Exception as error:
    st.error(f"An error occurred: {error}")
    st.info("Please check that Iris.csv is present in the repository.")
