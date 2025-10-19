import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

# === File path ===
file_path = r"C:\Users\abanu\Documents\swiggy.csv"

# === Step 1: Load dataset ===
df = pd.read_csv(file_path)
print("Original shape:", df.shape)

# === Step 2: Drop duplicates & handle missing ===
df.drop_duplicates(inplace=True)
df.dropna(subset=["name", "city", "rating", "cost", "cuisine"], inplace=True)

# === Step 3: Clean cost column ===
df["cost"] = df["cost"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
df["cost"].fillna(df["cost"].median(), inplace=True)

# === Step 4: Save cleaned data ===
df.to_csv("cleaned_data.csv", index=False)
print("Saved: cleaned_data.csv")

# === Step 5: One-Hot Encode only city (small category set) ===
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_city = encoder.fit_transform(df[["city"]])
encoded_city_df = pd.DataFrame(
    encoded_city, columns=encoder.get_feature_names_out(["city"])
)

# Combine with numeric columns only
numeric_cols = ["rating", "rating_count", "cost"]
final_df = pd.concat([encoded_city_df, df[numeric_cols].reset_index(drop=True)], axis=1)

# === Step 6: Save encoded data and encoder ===
final_df.to_csv("encoded_data.csv", index=False)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Saved: encoded_data.csv and encoder.pkl")
print("Data preparation complete (lightweight version)!")
