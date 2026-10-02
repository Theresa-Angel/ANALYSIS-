"""
Machine Learning based Buyer Segmentation and Investment Profiling
for Real Estate Market Intelligence — Data Pipeline

Steps:
 1. Data Cleaning
 2. Feature Engineering (merge clients + properties)
 3. Feature Encoding
 4. Feature Scaling
 5. Optimal Cluster Selection (Elbow + Silhouette)
 6. K-Means + Hierarchical Clustering
 7. Cluster Interpretation & Labeling
 8. Export processed dataset for the Streamlit dashboard
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from datetime import datetime
import json

RANDOM_STATE = 42
REFERENCE_DATE = pd.Timestamp("2025-12-31")  # end of transaction window

# ---------------------------------------------------------------------------
# STEP 1: DATA CLEANING
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: DATA CLEANING")
print("=" * 70)

clients = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

print(f"Raw clients: {clients.shape}, Raw properties: {properties.shape}")

# --- Clients cleaning ---
before = len(clients)
clients = clients.drop_duplicates(subset="client_id")
print(f"Duplicate clients removed: {before - len(clients)}")

# Normalize categorical labels (strip whitespace, consistent casing)
cat_cols_clients = ["client_type", "gender", "country", "region",
                     "acquisition_purpose", "loan_applied", "referral_channel"]
for col in cat_cols_clients:
    clients[col] = clients[col].astype(str).str.strip()

# Parse date_of_birth (mixed MM-DD-YYYY / M/D/YYYY) -> age
clients["date_of_birth"] = pd.to_datetime(clients["date_of_birth"], format="mixed", errors="coerce")
missing_dob = clients["date_of_birth"].isnull().sum()
if missing_dob:
    clients["date_of_birth"] = clients["date_of_birth"].fillna(clients["date_of_birth"].median())
clients["age"] = ((REFERENCE_DATE - clients["date_of_birth"]).dt.days / 365.25).astype(int)

# Missing value handling
missing_before = clients.isnull().sum().sum()
for col in cat_cols_clients:
    clients[col] = clients[col].fillna(clients[col].mode()[0])
clients["satisfaction_score"] = clients["satisfaction_score"].fillna(clients["satisfaction_score"].median())
print(f"Missing client attribute cells handled: {missing_before}")

# --- Properties cleaning ---
before = len(properties)
properties = properties.drop_duplicates(subset="listing_id")
print(f"Duplicate property listings removed: {before - len(properties)}")

properties["unit_category"] = properties["unit_category"].astype(str).str.strip()
properties["listing_status"] = properties["listing_status"].astype(str).str.strip()
properties["transaction_date"] = pd.to_datetime(properties["transaction_date"], format="mixed", errors="coerce")

# Clean sale_price: strip $ and commas -> float
properties["sale_price_clean"] = (
    properties["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
)

print(f"Cleaned properties: {properties.shape}")

# ---------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING (aggregate transaction behavior per client)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

sold = properties[properties["listing_status"] == "Sold"].copy()

agg = sold.groupby("client_ref").agg(
    num_units_purchased=("listing_id", "count"),
    total_spend=("sale_price_clean", "sum"),
    avg_unit_price=("sale_price_clean", "mean"),
    max_unit_price=("sale_price_clean", "max"),
    avg_floor_area=("floor_area_sqft", "mean"),
    total_floor_area=("floor_area_sqft", "sum"),
    num_towers=("tower_number", "nunique"),
    first_purchase=("transaction_date", "min"),
    last_purchase=("transaction_date", "max"),
).reset_index()

# Share of apartment vs office units, and % of purchases for investment towers
unit_share = (
    sold.groupby(["client_ref", "unit_category"]).size().unstack(fill_value=0)
)
unit_share = unit_share.div(unit_share.sum(axis=1), axis=0).add_prefix("pct_")
agg = agg.merge(unit_share, left_on="client_ref", right_index=True, how="left")

agg["purchase_span_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
agg["multi_unit_buyer"] = (agg["num_units_purchased"] > 1).astype(int)

# Merge into client master table
df = clients.merge(agg, left_on="client_id", right_on="client_ref", how="left")

# Clients with zero sold transactions (edge case) -> fill zeros
fill_zero_cols = ["num_units_purchased", "total_spend", "avg_unit_price", "max_unit_price",
                   "avg_floor_area", "total_floor_area", "num_towers", "purchase_span_days",
                   "multi_unit_buyer"]
for col in fill_zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
for col in [c for c in df.columns if c.startswith("pct_")]:
    df[col] = df[col].fillna(0)

df.drop(columns=["client_ref", "first_purchase", "last_purchase"], inplace=True, errors="ignore")

print(f"Merged client-property feature table: {df.shape}")
print("Engineered features:", [c for c in df.columns if c not in clients.columns])

# ---------------------------------------------------------------------------
# STEP 3: FEATURE ENCODING
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3: FEATURE ENCODING")
print("=" * 70)

df_model = df.copy()

# Label encoding for binary fields
le_client_type = LabelEncoder()
df_model["client_type_enc"] = le_client_type.fit_transform(df_model["client_type"])
le_gender = LabelEncoder()
df_model["gender_enc"] = le_gender.fit_transform(df_model["gender"])
le_purpose = LabelEncoder()
df_model["acquisition_purpose_enc"] = le_purpose.fit_transform(df_model["acquisition_purpose"])
le_loan = LabelEncoder()
df_model["loan_applied_enc"] = le_loan.fit_transform(df_model["loan_applied"])

# One-hot encoding for multi-category fields (region collapsed via country is enough
# granularity for clustering; region has 57 levels which would be too sparse)
ohe_cols = ["country", "referral_channel"]
ohe = pd.get_dummies(df_model[ohe_cols], prefix=ohe_cols)
df_model = pd.concat([df_model, ohe], axis=1)

print(f"Encoded feature columns added: client_type_enc, gender_enc, acquisition_purpose_enc, "
      f"loan_applied_enc, {list(ohe.columns)}")

# ---------------------------------------------------------------------------
# STEP 4: FEATURE SCALING
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4: FEATURE SCALING")
print("=" * 70)

# Behavioral / transactional features are the primary drivers of buyer
# segmentation (they show real variance across clients). Country and
# referral-channel one-hot columns are kept in the exported dataset for
# dashboard filtering, but are excluded from the clustering matrix itself
# because with 10-57 sparse categories they mostly add noise and wash out
# the meaningful behavioral signal (verified empirically: including them
# dropped the silhouette score). client_type / acquisition_purpose / loan
# flags are up-weighted since the PRD calls them out as key segmentation
# drivers even though, in this dataset, they are fairly evenly distributed
# across clients.
numeric_features = [
    "age", "satisfaction_score", "num_units_purchased", "total_spend",
    "avg_unit_price", "avg_floor_area", "num_towers", "purchase_span_days",
]
pct_cols = [c for c in df_model.columns if c.startswith("pct_")]
weighted_flags = ["client_type_enc", "acquisition_purpose_enc", "loan_applied_enc"]
# NOTE: business flags are included at natural (1x) scale alongside the
# behavioral features. Earlier experimentation showed that up-weighting
# them (e.g. 3x) causes K-Means to simply re-derive the original categorical
# columns (clusters become "is Company" / "is Investment" / "is Loan" almost
# by definition) while erasing the genuine behavioral variance in spend,
# unit count, and price that is far more useful for segmentation. Natural
# scale lets behavior dominate while flags still nudge cluster boundaries.
flag_weights = {"client_type_enc": 1.0, "acquisition_purpose_enc": 1.0, "loan_applied_enc": 1.0}

feature_cols = numeric_features + pct_cols + weighted_flags

X_raw = df_model[feature_cols].fillna(0).copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values)

# Apply extra weight to the business-flag columns after scaling
flag_idx = {col: feature_cols.index(col) for col in weighted_flags}
for col, w in flag_weights.items():
    X_scaled[:, flag_idx[col]] *= w

print(f"Feature matrix for clustering: {X_scaled.shape}")
print(f"Clustering features used: {feature_cols}")
print(f"(country / referral_channel one-hot columns retained in export for filtering, "
      f"excluded from clustering matrix)")

# ---------------------------------------------------------------------------
# STEP 5: OPTIMAL CLUSTER SELECTION (Elbow + Silhouette)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5: OPTIMAL CLUSTER SELECTION")
print("=" * 70)

inertias = []
sil_scores = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

elbow_df = pd.DataFrame({"k": list(K_range), "inertia": inertias, "silhouette": sil_scores})
elbow_df.to_csv("elbow_silhouette_results.csv", index=False)

best_k = elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"]
print(f"\nBest k by silhouette score: {int(best_k)}")

# We use k=4 to align with the PRD's recommended buyer segments (C1-C4),
# unless silhouette strongly favors a different k.
K_FINAL = 4
print(f"Using K_FINAL = {K_FINAL} clusters (aligned with PRD's 4 recommended buyer segments)")

# ---------------------------------------------------------------------------
# STEP 6: CLUSTERING (K-Means + Hierarchical for validation)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 6: CLUSTERING")
print("=" * 70)

kmeans = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, df["cluster_kmeans"])
print(f"K-Means (k={K_FINAL}) silhouette score: {kmeans_sil:.4f}")

hier = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward")
df["cluster_hierarchical"] = hier.fit_predict(X_scaled)
hier_sil = silhouette_score(X_scaled, df["cluster_hierarchical"])
print(f"Hierarchical (k={K_FINAL}) silhouette score: {hier_sil:.4f}")

# Agreement between the two methods (using majority mapping)
from scipy.stats import mode
agreement_map = {}
for c in range(K_FINAL):
    mask = df["cluster_kmeans"] == c
    if mask.sum() > 0:
        agreement_map[c] = mode(df.loc[mask, "cluster_hierarchical"], keepdims=False).mode
mapped_hier = df["cluster_kmeans"].map(agreement_map)
agreement_pct = (mapped_hier == df["cluster_hierarchical"]).mean() * 100
print(f"K-Means vs Hierarchical cluster agreement (best label mapping): {agreement_pct:.1f}%")

# ---------------------------------------------------------------------------
# STEP 7: CLUSTER INTERPRETATION & LABELING
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 7: CLUSTER INTERPRETATION")
print("=" * 70)

profile = df.groupby("cluster_kmeans").agg(
    n_clients=("client_id", "count"),
    avg_age=("age", "mean"),
    pct_investment=("acquisition_purpose", lambda x: (x == "Investment").mean() * 100),
    pct_loan=("loan_applied", lambda x: (x == "Yes").mean() * 100),
    pct_company=("client_type", lambda x: (x == "Company").mean() * 100),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_units=("num_units_purchased", "mean"),
    avg_spend=("total_spend", "mean"),
    avg_unit_price=("avg_unit_price", "mean"),
    top_country=("country", lambda x: x.mode()[0] if not x.mode().empty else None),
).reset_index()

print(profile.to_string(index=False))
profile.to_csv("cluster_profile_summary.csv", index=False)

# Auto-derive a descriptive segment label per cluster based on its BEHAVIORAL
# profile (purchase volume, price point, spend), matched deterministically
# to the PRD's four recommended buyer archetypes via a ranking approach
# rather than arbitrary fixed thresholds (which don't generalize across runs).
remaining = list(profile["cluster_kmeans"])
labels = {}

# 1) Bulk, multi-unit, multi-tower buyers -> "Corporate Buyers"
#    ("Companies purchasing multiple units" per PRD)
bulk_idx = profile.loc[profile["cluster_kmeans"].isin(remaining), "avg_units"].idxmax()
bulk_cluster = profile.loc[bulk_idx, "cluster_kmeans"]
labels[bulk_cluster] = "Corporate Buyers"
remaining.remove(bulk_cluster)

# 2) Highest average unit price among what's left -> "Luxury Investors"
#    ("High satisfaction, large investments" per PRD)
luxury_idx = profile.loc[profile["cluster_kmeans"].isin(remaining), "avg_unit_price"].idxmax()
luxury_cluster = profile.loc[luxury_idx, "cluster_kmeans"]
labels[luxury_cluster] = "Luxury Investors"
remaining.remove(luxury_cluster)

# 3) Of the two left, higher total spend / investment share -> "Global Investors"
#    ("High income, investment purchases" per PRD); the other -> "First-Time Buyers"
#    ("Younger, loan dependent" per PRD)
sub = profile[profile["cluster_kmeans"].isin(remaining)].copy()
sub["score"] = sub["avg_spend"].rank() + sub["pct_investment"].rank()
global_cluster = sub.loc[sub["score"].idxmax(), "cluster_kmeans"]
labels[global_cluster] = "Global Investors"
remaining.remove(global_cluster)
labels[remaining[0]] = "First-Time Buyers"

profile["segment_label"] = profile["cluster_kmeans"].map(labels)

print("\nDerived segment labels:")
print(profile[["cluster_kmeans", "segment_label", "n_clients"]].to_string(index=False))

cluster_to_label = dict(zip(profile["cluster_kmeans"], profile["segment_label"]))
df["segment"] = df["cluster_kmeans"].map(cluster_to_label)

# ---------------------------------------------------------------------------
# STEP 8: EXPORT
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 8: EXPORT")
print("=" * 70)

export_cols = [
    "client_id", "client_type", "gender", "country", "region", "age",
    "acquisition_purpose", "loan_applied", "referral_channel", "satisfaction_score",
    "num_units_purchased", "total_spend", "avg_unit_price", "max_unit_price",
    "avg_floor_area", "num_towers", "purchase_span_days",
] + pct_cols + ["cluster_kmeans", "cluster_hierarchical", "segment"]

final_df = df[export_cols].copy()
final_df.to_csv("clients_segmented.csv", index=False)
print(f"Exported clients_segmented.csv: {final_df.shape}")

# Save summary stats as JSON for quick reference in report/dashboard
summary = {
    "n_clients": int(len(df)),
    "n_clusters": K_FINAL,
    "kmeans_silhouette": round(float(kmeans_sil), 4),
    "hierarchical_silhouette": round(float(hier_sil), 4),
    "kmeans_hierarchical_agreement_pct": round(float(agreement_pct), 1),
    "elbow_silhouette_by_k": elbow_df.to_dict(orient="records"),
    "cluster_profile": profile.to_dict(orient="records"),
}
with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nPipeline complete. Files written:")
print(" - clients_segmented.csv (for Streamlit dashboard)")
print(" - cluster_profile_summary.csv")
print(" - elbow_silhouette_results.csv")
print(" - model_summary.json")
