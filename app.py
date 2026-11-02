from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import shap
import streamlit as st


MODEL_PATH = Path(__file__).parent / "artifacts" / "colombo_portcalls_model.joblib"
FEATURES = ["lag_1", "lag_7", "avg_7", "month"]


@st.cache_resource
def load_model_artifact(artifact_path: str, artifact_modified: float):
    """Load the fitted notebook model and prepare its SHAP explainer."""
    del artifact_modified  # Included in the cache key so new artifacts reload.
    artifact = joblib.load(artifact_path)
    required_keys = {
        "model",
        "features",
        "history",
        "background",
        "best_alpha",
        "validation_mae",
    }
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required values: {', '.join(sorted(missing_keys))}"
        )

    history = artifact["history"].copy()
    features = artifact["features"]
    if features != FEATURES:
        raise ValueError(f"Artifact features do not match the app: {features}")
    if not {"date", "portcalls"}.issubset(history.columns) or history.empty:
        raise ValueError("Model artifact must contain non-empty date and portcalls history.")

    explainer = shap.Explainer(
        artifact["model"].predict,
        shap.maskers.Independent(artifact["background"]),
        algorithm="permutation",
    )
    return (
        history,
        artifact["model"],
        explainer,
        artifact["best_alpha"],
        artifact["validation_mae"],
    )


st.set_page_config(
    page_title="Colombo Port Activity Forecast",
    page_icon="⚓",
    layout="wide",
)

st.title("Colombo Port Activity Forecast")
st.caption(
    "Forecast daily vessel calls with a Poisson regression model and inspect "
    "feature contributions with SHAP."
)

if not MODEL_PATH.is_file():
    st.error(
        f"Could not find the trained model artifact: {MODEL_PATH}. "
        "Run the final export cell in SL_port_activity_prediction.ipynb first."
    )
    st.stop()

try:
    history, model, explainer, best_alpha, validation_mae = load_model_artifact(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Unable to load the trained model artifact: {error}")
    st.stop()

latest = history.iloc[-1]
latest_date = latest["date"].date()
forecast_default = latest_date + timedelta(days=1)
recent_week = history["portcalls"].iloc[-7:]

metric_cols = st.columns(3)
metric_cols[0].metric("Latest recorded activity", f"{int(latest['portcalls'])} calls")
metric_cols[1].metric("Latest data date", latest_date.strftime("%d %b %Y"))
metric_cols[2].metric("2024 validation MAE", f"{validation_mae:.2f} calls")

st.subheader("Set forecast inputs")
st.write(
    "Choose a forecast date and, if needed, adjust the recent activity values. "
    "The defaults are taken from the latest Colombo observations."
)

with st.form("forecast_form"):
    forecast_date = st.date_input(
        "Date to forecast",
        value=forecast_default,
        min_value=forecast_default,
    )
    input_cols = st.columns(3)
    lag_1 = input_cols[0].number_input(
        "Calls on the previous day (lag 1)",
        min_value=0,
        value=int(latest["portcalls"]),
        step=1,
    )
    lag_7 = input_cols[1].number_input(
        "Calls seven days before forecast (lag 7)",
        min_value=0,
        value=int(history["portcalls"].iloc[-7]),
        step=1,
    )
    avg_7 = input_cols[2].number_input(
        "Average calls over the previous 7 days",
        min_value=0.0,
        value=float(recent_week.mean()),
        step=0.1,
    )
    submitted = st.form_submit_button("Generate forecast", type="primary")

st.subheader("Recent Colombo activity")
fig, ax = plt.subplots(figsize=(11, 3.5))
recent_history = history.tail(90)
ax.plot(recent_history["date"], recent_history["portcalls"], color="#168aad", linewidth=1.5)
ax.set_ylabel("Vessel calls")
ax.set_xlabel("Date")
ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)
plt.close(fig)

if submitted:
    features = pd.DataFrame(
        [{
            "lag_1": lag_1,
            "lag_7": lag_7,
            "avg_7": avg_7,
            "month": forecast_date.month,
        }],
        columns=FEATURES,
    )
    prediction = float(model.predict(features)[0])

    result_cols = st.columns(2)
    result_cols[0].metric(
        f"Expected calls on {forecast_date.strftime('%d %b %Y')}",
        f"{prediction:.2f}",
    )
    result_cols[1].metric("Rounded estimate", f"{round(prediction)} calls")

    st.subheader("Why did the model make this prediction?")
    explanation = explainer(
        features,
        max_evals=2 * len(FEATURES) + 1,
    )
    shap.plots.waterfall(explanation[0], show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close()

    contributions = pd.DataFrame({
        "Feature": FEATURES,
        "Input value": [features.iloc[0][feature] for feature in FEATURES],
        "Contribution to prediction": explanation.values[0],
    }).sort_values("Contribution to prediction", key=abs, ascending=False)
    st.dataframe(contributions, hide_index=True, width="stretch")
    st.caption(
        "Positive SHAP values push the prediction above the model baseline; "
        "negative values push it below. Contributions are in vessel-call units."
    )

with st.expander("Model details"):
    st.write(
        f"The model is a StandardScaler + PoissonRegressor pipeline. "
        f"Regularization (alpha={best_alpha:g}) was selected using 2024 validation "
        "MAE, then the selected model was fit on all available Colombo training "
        "data stored in the notebook-generated artifact."
    )
    st.write(
        "Features: previous-day calls, calls seven days earlier, the preceding "
        "7-day average, and the forecast month."
    )
