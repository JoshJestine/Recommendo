"""
Streamlit app: given a driver profile, recommend the cheapest vehicles to insure.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cheapest Cars to Insure",
    page_icon="🚗",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"

# -----------------------------------------------------------------------------
# Defaults for fields the user does NOT provide
# -----------------------------------------------------------------------------
# These are "best-case" defaults — they produce the lowest-baseline prediction,
# so what the user sees is the floor for their profile. Change to taste.
DEFAULTS = {
    "Gender": "Female",
    "Model_Year": 2022,
    "Ownership": "Owned",
    "Previous_Claims": 0,
}

# Credit-tier → representative credit score (mid-range of each FICO band)
CREDIT_TIERS = {
    "Excellent (800-850)": 820,
    "Very Good (740-799)": 770,
    "Good (670-739)": 700,
    "Fair (580-669)": 620,
    "Poor (300-579)": 540,
}


# -----------------------------------------------------------------------------
# Artifact loading (cached so the model is loaded once)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not (ART_DIR / "model.json").exists():
        st.error(
            "❌ Trained model not found. Run `python train.py` first."
        )
        st.stop()

    model = xgb.XGBRegressor(enable_categorical=True)
    model.load_model(ART_DIR / "model.json")

    with open(ART_DIR / "feature_schema.json") as f:
        schema = json.load(f)

    catalog = pd.read_csv(ART_DIR / "vehicle_catalog.csv")

    with open(ART_DIR / "state_list.json") as f:
        states = json.load(f)

    return model, schema, catalog, states


def build_inference_frame(profile: dict, catalog: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Replicate the user's profile across every vehicle in the catalog so we
    can score them all in a single batched predict() call."""
    n = len(catalog)
    rows = {col: [profile[col]] * n for col in profile}
    rows["Make"] = catalog["Make"].tolist()
    rows["Model"] = catalog["Model"].tolist()
    df = pd.DataFrame(rows)

    # Restore column order used at training time
    df = df[schema["feature_order"]]

    # Match dtypes — categorical columns must use the SAME category levels
    # the model was trained on, otherwise XGBoost will error or misalign.
    for col, levels in schema["categorical"].items():
        df[col] = pd.Categorical(df[col].astype(str), categories=levels)
    for col in schema["numeric"]:
        df[col] = pd.to_numeric(df[col])

    return df


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
model, schema, catalog, states = load_artifacts()

st.title("🚗 Cheapest Cars to Insure")
st.caption(
    "Enter your profile and get the vehicles with the lowest predicted monthly "
    "full-coverage premium. Predictions use an XGBoost model trained on a "
    "synthetic dataset grounded in published 2025-2026 industry rate data."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age",
        min_value=16,
        max_value=99,
        value=35,
        step=1,
        help="Your age in years.",
    )
    state = st.selectbox(
        "State",
        options=states,
        index=states.index("California") if "California" in states else 0,
    )
with col2:
    location = st.selectbox(
        "Where do you live?",
        options=["Urban", "Suburban", "Rural"],
        index=1,
    )
    credit_tier = st.selectbox(
        "Credit tier",
        options=list(CREDIT_TIERS.keys()),
        index=2,  # Good
    )

n_results = st.slider("How many recommendations?", 3, 15, 5)

st.divider()

if st.button("Find cheapest vehicles to insure", type="primary", use_container_width=True):
    profile = {
        "Age": age,
        "Gender": DEFAULTS["Gender"],
        "State": state,
        "Location": location,
        "Credit_Score": CREDIT_TIERS[credit_tier],
        "Model_Year": DEFAULTS["Model_Year"],
        "Ownership": DEFAULTS["Ownership"],
        "Previous_Claims": DEFAULTS["Previous_Claims"],
    }

    X_inf = build_inference_frame(profile, catalog, schema)
    preds = model.predict(X_inf)

    results = catalog.copy()
    results["Predicted_Monthly_$"] = preds.round(2)
    results["Predicted_Annual_$"]  = (preds * 12).round(0).astype(int)
    results = results.sort_values("Predicted_Monthly_$", ascending=True).reset_index(drop=True)

    cheapest = results.head(n_results).copy()
    cheapest.insert(0, "Rank", range(1, len(cheapest) + 1))

    avg_pred = float(preds.mean())
    min_pred = float(preds.min())
    max_pred = float(preds.max())

    st.success(f"Showing top {n_results} cheapest vehicles for your profile.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Cheapest in catalog",  f"${min_pred:,.0f}/mo")
    m2.metric("Average across catalog", f"${avg_pred:,.0f}/mo")
    m3.metric("Most expensive",       f"${max_pred:,.0f}/mo")

    st.subheader("🏆 Cheapest vehicles for you")
    st.dataframe(
        cheapest,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank":               st.column_config.NumberColumn(width="small"),
            "Predicted_Monthly_$": st.column_config.NumberColumn(format="$%.2f"),
            "Predicted_Annual_$":  st.column_config.NumberColumn(format="$%d"),
        },
    )

    with st.expander("Most expensive vehicles for your profile (for comparison)"):
        priciest = results.tail(n_results)[::-1].copy()
        priciest.insert(0, "Rank", range(1, len(priciest) + 1))
        st.dataframe(
            priciest,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted_Monthly_$": st.column_config.NumberColumn(format="$%.2f"),
                "Predicted_Annual_$":  st.column_config.NumberColumn(format="$%d"),
            },
        )

    with st.expander("What assumptions am I making for fields you didn't provide?"):
        st.write(
            "- **Gender:** Female (no impact in states that ban gender as a rating factor)\n"
            f"- **Model year:** {DEFAULTS['Model_Year']} (a typical recent vehicle)\n"
            f"- **Ownership:** {DEFAULTS['Ownership']}\n"
            f"- **Previous claims:** {DEFAULTS['Previous_Claims']} (clean record)\n\n"
            "Change these defaults in `app.py` (search for `DEFAULTS = {...}`) "
            "if you want to model a different baseline."
        )

st.divider()
st.caption(
    "⚠️ This is a leisure ML project. Predictions are illustrative and based on "
    "a synthetic dataset — they are **not** real insurance quotes. For an actual "
    "quote, contact an insurance carrier."
)
