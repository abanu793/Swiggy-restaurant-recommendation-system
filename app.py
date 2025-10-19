import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle


# ----------------- LOAD DATA -----------------
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


cleaned_df = load_csv("cleaned_data.csv")
encoded_df = load_csv("encoded_data.csv")
encoder = load_pickle("encoder.pkl")

# ----------------- CLEAN DATA -----------------
for col in ["city", "cuisine"]:
    if col in cleaned_df.columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.title()

# Clean cost
if "cost" in cleaned_df.columns:
    cleaned_df["cost"] = (
        cleaned_df["cost"]
        .astype(str)
        .str.replace("₹|,|for two|for one", "", regex=True)
        .str.extract(r"(\d+(\.\d+)?)")[0]
        .astype(float)
        .fillna(0)
    )
    if cleaned_df["cost"].quantile(0.95) > 5000:
        cleaned_df["cost"] /= 100
        st.info("Detected inflated cost values — auto-normalized.")

for col in ["rating", "rating_count"]:
    if col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce").fillna(0)

X_sparse = csr_matrix(encoded_df.apply(pd.to_numeric, errors="coerce").fillna(0).values)

# ----------------- AUTO-DETECT RESTAURANT COLUMN -----------------
restaurant_col = next(
    (
        c
        for c in ["restaurant_name", "Restaurant Name", "name", "Name"]
        if c in cleaned_df.columns
    ),
    None,
)
if restaurant_col is None:
    st.error("No restaurant column found!")
    st.stop()

# ----------------- STREAMLIT UI -----------------
st.title("Restaurant Recommendation System")
st.sidebar.header("Select Your Preferences")

selected_city = st.sidebar.selectbox(
    "Select City", options=cleaned_df["city"].sort_values().unique()
)
selected_cuisine = st.sidebar.selectbox(
    "Select Cuisine", options=cleaned_df["cuisine"].sort_values().unique()
)
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# ----------------- COST SLIDER -----------------
cost_min, cost_max = int(cleaned_df["cost"].min()), int(cleaned_df["cost"].max())
scale = 100 if cost_max > 5000 else 1
slider_val = st.sidebar.slider(
    "Maximum Cost",
    int(cost_min / scale),
    int(cost_max / scale),
    int(cost_max / scale),
    step=1,
    format="₹%d",
)
max_cost = slider_val * scale


# ----------------- FILTER DATA -----------------
def filter_restaurants(df):
    filt = df[
        (df["city"] == selected_city)
        & (df["cuisine"] == selected_cuisine)
        & (df["rating"] >= min_rating)
        & (df["cost"] <= max_cost)
    ]
    if filt.empty:
        filt = df[(df["city"] == selected_city) & (df["cuisine"] == selected_cuisine)]
    if filt.empty:
        filt = df[df["city"] == selected_city]
    return filt


filtered_df = filter_restaurants(cleaned_df)
st.markdown(f"**Total restaurants matching filters:** {filtered_df.shape[0]}")


# ----------------- TABLE DISPLAY -----------------
def display_table(df):
    df = df.copy()
    if "cost" in df.columns:
        df["cost"] = df["cost"].apply(lambda x: f"₹{int(x)}")
    if "link" in df.columns:
        df[restaurant_col] = df.apply(
            lambda r: (
                f"[{r[restaurant_col]}]({r['link']})"
                if pd.notna(r.get("link")) and str(r["link"]).startswith("http")
                else r[restaurant_col]
            ),
            axis=1,
        )
    cols = [
        c
        for c in [
            restaurant_col,
            "city",
            "cuisine",
            "rating",
            "rating_count",
            "cost",
            "address",
            "link",
        ]
        if c in df.columns
    ]
    st.markdown(
        df[cols].reset_index(drop=True).to_markdown(index=False), unsafe_allow_html=True
    )


display_table(filtered_df)

# ----------------- RECOMMENDATIONS -----------------
filtered_indices = filtered_df.index.tolist()
if len(filtered_indices) < 2:
    recommended_restaurants = filtered_df
else:
    sim = cosine_similarity(X_sparse[filtered_indices], X_sparse[filtered_indices])
    scores = sim[0].flatten()
    top_idx = np.argsort(scores)[::-1][1 : top_n + 1]
    recommended_restaurants = filtered_df.iloc[top_idx]

st.subheader("Recommended Restaurants")
display_table(recommended_restaurants)
