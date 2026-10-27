from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import fetch_california_housing


st.set_page_config(
    page_title="California Housing Market Analysis",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def load_housing_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.copy()
    frame["MedHouseValUsd"] = frame["MedHouseVal"] * 100_000
    return frame


@st.cache_resource
def load_prediction_model():
    project_dir = Path(__file__).resolve().parent
    model = joblib.load(project_dir / "house_model.joblib")
    features = joblib.load(project_dir / "houe_features.joblib")
    return model, features


st.title("California Housing Market Analysis")
st.markdown(
    "Explore how block-group housing values relate to household income, "
    "housing characteristics, and location."
)

data = load_housing_data()

with st.sidebar:
    st.header("Filter the data")
    income_range = st.slider(
        "Median block-group income (in $10,000s)",
        min_value=float(data["MedInc"].min()),
        max_value=float(data["MedInc"].max()),
        value=(float(data["MedInc"].min()), float(data["MedInc"].max())),
        step=0.1,
    )
    age_range = st.slider(
        "Median house age (years)",
        min_value=int(data["HouseAge"].min()),
        max_value=int(data["HouseAge"].max()),
        value=(int(data["HouseAge"].min()), int(data["HouseAge"].max())),
    )

filtered = data.loc[
    data["MedInc"].between(*income_range)
    & data["HouseAge"].between(*age_range)
].copy()

if filtered.empty:
    st.warning("No block groups match these filters. Expand the ranges to see results.")
    st.stop()

median_value = filtered["MedHouseValUsd"].median()
median_income_usd = filtered["MedInc"].median() * 10_000
income_value_correlation = filtered["MedInc"].corr(filtered["MedHouseVal"])
correlation_text = (
    "not available"
    if pd.isna(income_value_correlation)
    else f"{income_value_correlation:.2f}"
)

metric_columns = st.columns(3)
metric_columns[0].metric("Block groups", f"{len(filtered):,}")
metric_columns[1].metric("Median block-group home value", f"${median_value:,.0f}")
metric_columns[2].metric("Median household income", f"${median_income_usd:,.0f}")

st.caption(
    "Values describe California census block groups in the 1990 census dataset; "
    "they are not current listings or individual home sale prices."
)

left_chart, right_chart = st.columns(2)

with left_chart:
    distribution = px.histogram(
        filtered,
        x="MedHouseValUsd",
        nbins=35,
        title="Distribution of block-group median home values",
        labels={"MedHouseValUsd": "Median home value (USD)", "count": "Block groups"},
        color_discrete_sequence=["#2a6f97"],
    )
    distribution.update_layout(yaxis_title="Block groups")
    st.plotly_chart(distribution, width="stretch")

with right_chart:
    income_sample = filtered.sample(
        n=min(len(filtered), 5_000),
        random_state=42,
    )
    income_scatter = px.scatter(
        income_sample,
        x="MedInc",
        y="MedHouseValUsd",
        color="HouseAge",
        color_continuous_scale="Viridis",
        title="Income and home value",
        labels={
            "MedInc": "Median income (in $10,000s)",
            "MedHouseValUsd": "Median home value (USD)",
            "HouseAge": "Median house age (years)",
        },
        opacity=0.6,
    )
    st.plotly_chart(income_scatter, width="stretch")

st.subheader("Average value by income group")
income_group_data = filtered[["MedInc", "MedHouseValUsd"]].copy()
if income_group_data["MedInc"].nunique() == 1:
    income_group_data["Income group"] = "All selected"
else:
    income_bins = pd.qcut(
        income_group_data["MedInc"],
        q=4,
        duplicates="drop",
    )
    income_group_data["Income group"] = income_bins.cat.rename_categories(
        income_bins.cat.categories.astype(str)
    )
income_group_summary = (
    income_group_data.groupby("Income group", observed=True)["MedHouseValUsd"]
    .mean()
    .reset_index(name="Average block-group home value (USD)")
)
income_group_chart = px.bar(
    income_group_summary,
    x="Income group",
    y="Average block-group home value (USD)",
    title="Average home value across income groups",
    labels={"Income group": "Median income group"},
    color="Average block-group home value (USD)",
    color_continuous_scale="Blues",
)
income_group_chart.update_layout(coloraxis_showscale=False)
st.plotly_chart(income_group_chart, width="stretch")

st.subheader("Geographic pattern")
map_sample = filtered.sample(n=min(len(filtered), 5_000), random_state=42)
map_chart = px.scatter_geo(
    map_sample,
    lat="Latitude",
    lon="Longitude",
    color="MedHouseValUsd",
    color_continuous_scale="Plasma",
    hover_data={
        "MedInc": ":.1f",
        "HouseAge": True,
        "MedHouseValUsd": ":$,.0f",
        "Latitude": False,
        "Longitude": False,
    },
    labels={
        "MedHouseValUsd": "Median home value (USD)",
        "MedInc": "Median income (in $10,000s)",
        "HouseAge": "Median house age (years)",
    },
    title="Block-group home values across California",
)
map_chart.update_geos(
    scope="usa",
    fitbounds="locations",
    showcountries=True,
    showsubunits=True,
)
map_chart.update_layout(height=550)
st.plotly_chart(map_chart, width="stretch")

st.subheader("Analyst notes")
lowest_income_value = income_group_summary.iloc[0]["Average block-group home value (USD)"]
highest_income_value = income_group_summary.iloc[-1]["Average block-group home value (USD)"]
note_columns = st.columns(2)
note_columns[0].write(
    f"Within the selected data, the correlation between median income and "
    f"block-group home value is **{correlation_text}**. "
    "This is an association, not evidence that income causes home values to change."
)
note_columns[1].write(
    f"Average block-group value ranges from **${lowest_income_value:,.0f}** "
    f"in the lowest income group to **${highest_income_value:,.0f}** "
    "in the highest income group. Groups are recalculated after filtering."
)

st.info(
    "Interpretation limits: the dataset is historical, aggregates households by "
    "block group, and caps the recorded target at $500,000. Treat these findings "
    "as an example of exploratory analysis, not a current market valuation."
)

st.divider()
st.header("Estimate a block-group home value")
st.markdown(
    "Enter the same block-group characteristics used to train the existing "
    "prediction model."
)

with st.form("home_prediction_form"):
    input_columns = st.columns(2)
    with input_columns[0]:
        median_income = st.number_input(
            "Median income (in $10,000s)",
            min_value=0.0001,
            value=float(data["MedInc"].median()),
            step=0.1,
            help="For example, enter 5 for a median income of about $50,000.",
        )
        house_age = st.number_input(
            "Median house age (years)",
            min_value=0.0,
            value=float(data["HouseAge"].median()),
            step=1.0,
        )
        average_rooms = st.number_input(
            "Average rooms per household",
            min_value=0.0001,
            value=float(data["AveRooms"].median()),
            step=0.1,
        )
        average_bedrooms = st.number_input(
            "Average bedrooms per household",
            min_value=0.0001,
            value=float(data["AveBedrms"].median()),
            step=0.1,
        )
    with input_columns[1]:
        population = st.number_input(
            "Block-group population",
            min_value=0.0001,
            value=float(data["Population"].median()),
            step=10.0,
        )
        average_occupancy = st.number_input(
            "Average household occupancy",
            min_value=0.0001,
            value=float(data["AveOccup"].median()),
            step=0.1,
        )
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=float(data["Latitude"].median()),
            step=0.01,
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=float(data["Longitude"].median()),
            step=0.01,
        )
    submitted = st.form_submit_button("Estimate value")

if submitted:
    try:
        prediction_model, model_features = load_prediction_model()
        prediction_input = pd.DataFrame(
            [
                {
                    "MedInc": median_income,
                    "HouseAge": house_age,
                    "AveRooms": average_rooms,
                    "AveBedrms": average_bedrooms,
                    "Population": population,
                    "AveOccup": average_occupancy,
                    "Latitude": latitude,
                    "Longitude": longitude,
                }
            ],
            columns=model_features,
        )
        predicted_value = float(prediction_model.predict(prediction_input)[0]) * 100_000
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not make a prediction: {error}")
    else:
        st.metric("Estimated block-group median value", f"${predicted_value:,.0f}")
        st.caption(
            "This is a model estimate based on historical census block-group data, "
            "not a current property appraisal or guaranteed sale price."
        )
