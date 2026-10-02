"""
Buyer Segmentation & Investment Profiling Dashboard
Real Estate Market Intelligence — Parcl Co. Limited x Unified Mentor
"""

import json
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths relative to this script so the app works regardless of
# the working directory (local or Streamlit Cloud)
_HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Parcl Buyer Segmentation & Investment Profiling",
    page_icon="\U0001F3E2",
    layout="wide",
)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(_HERE, "clients_segmented.csv"))
    with open(os.path.join(_HERE, "model_summary.json")) as f:
        summary = json.load(f)
    return df, summary

df, summary = load_data()

SEGMENT_COLORS = {
    "Global Investors": "#2563eb",
    "First-Time Buyers": "#16a34a",
    "Corporate Buyers": "#d97706",
    "Luxury Investors": "#7c3aed",
}
SEGMENT_DESCRIPTIONS = {
    "Global Investors": "High investment share and above-average spend — internationally minded buyers targeting appreciation.",
    "First-Time Buyers": "Lower average spend and unit price — often financing-dependent, entry-level buyers.",
    "Luxury Investors": "Highest average unit price and floor area — fewer units, but premium price points and satisfaction.",
    "Corporate Buyers": "Small in number but purchase many units across many towers — bulk / institutional buying pattern.",
}

# ---------------------------------------------------------------------------
# SIDEBAR — GLOBAL FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F3E2 Parcl Market Intelligence")
st.sidebar.caption("Buyer Segmentation & Investment Profiling")
st.sidebar.markdown("---")
st.sidebar.header("Filters")

countries = sorted(df["country"].unique())
sel_countries = st.sidebar.multiselect("Country", countries, default=[])

regions_available = sorted(df[df["country"].isin(sel_countries)]["region"].unique()) if sel_countries else sorted(df["region"].unique())
sel_regions = st.sidebar.multiselect("Region", regions_available, default=[])

sel_purpose = st.sidebar.multiselect("Acquisition Purpose", sorted(df["acquisition_purpose"].unique()), default=[])
sel_client_type = st.sidebar.multiselect("Client Type", sorted(df["client_type"].unique()), default=[])
sel_segment = st.sidebar.multiselect("Buyer Segment", sorted(df["segment"].unique()), default=[])

filtered = df.copy()
if sel_countries:
    filtered = filtered[filtered["country"].isin(sel_countries)]
if sel_regions:
    filtered = filtered[filtered["region"].isin(sel_regions)]
if sel_purpose:
    filtered = filtered[filtered["acquisition_purpose"].isin(sel_purpose)]
if sel_client_type:
    filtered = filtered[filtered["client_type"].isin(sel_client_type)]
if sel_segment:
    filtered = filtered[filtered["segment"].isin(sel_segment)]

st.sidebar.markdown("---")
st.sidebar.metric("Clients in view", f"{len(filtered):,}", delta=f"of {len(df):,} total")
st.sidebar.markdown(
    f"**Model quality**\n\n"
    f"K-Means silhouette: `{summary['kmeans_silhouette']}`\n\n"
    f"Hierarchical silhouette: `{summary['hierarchical_silhouette']}`\n\n"
    f"Method agreement: `{summary['kmeans_hierarchical_agreement_pct']}%`"
)

if filtered.empty:
    st.warning("No clients match the selected filters. Please broaden your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("Buyer Segmentation and Investment Profiling")
st.caption("Machine learning based buyer intelligence for real estate market strategy — Parcl Co. Limited")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Clients", f"{len(filtered):,}")
k2.metric("Total Spend", f"${filtered['total_spend'].sum()/1e6:,.1f}M")
k3.metric("Avg. Unit Price", f"${filtered['avg_unit_price'].mean():,.0f}")
k4.metric("Avg. Satisfaction", f"{filtered['satisfaction_score'].mean():.2f} / 5")
k5.metric("Segments Present", f"{filtered['segment'].nunique()}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001F4CA Buyer Segmentation Overview",
    "\U0001F4B0 Investor Behavior Dashboard",
    "\U0001F30D Geographic Buyer Analysis",
    "\U0001F4CB Segment Insights Panel",
])

# ---------------------------------------------------------------------------
# TAB 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Cluster Distribution")
    col1, col2 = st.columns([1, 1])

    with col1:
        seg_counts = filtered["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(
            seg_counts, names="segment", values="count", hole=0.45,
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Client Distribution by Buyer Segment",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            seg_counts.sort_values("count"), x="count", y="segment", orientation="h",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Segment Size (Number of Clients)", text="count",
        )
        fig2.update_layout(showlegend=False, yaxis_title="", xaxis_title="Clients")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cluster Validation: K-Means vs. Hierarchical")
    c1, c2 = st.columns(2)
    with c1:
        elbow_df = pd.DataFrame(summary["elbow_silhouette_by_k"])
        fig3 = px.line(elbow_df, x="k", y="inertia", markers=True, title="Elbow Method — Inertia vs. k")
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        fig4 = px.line(elbow_df, x="k", y="silhouette", markers=True, title="Silhouette Score vs. k")
        fig4.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 selected")
        st.plotly_chart(fig4, use_container_width=True)

    st.caption(
        "Four clusters were selected to align with the four buyer archetypes the business needs to act on "
        "(Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors), even though the silhouette "
        "curve does not peak sharply at k=4 — a common trade-off between statistical optimum and business "
        "interpretability."
    )

    st.subheader("Segment Composition Cross-tab")
    cross = pd.crosstab(filtered["segment"], filtered["acquisition_purpose"], normalize="index") * 100
    fig5 = px.imshow(
        cross, text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Acquisition Purpose", y="Segment", color="% of segment"),
        title="Acquisition Purpose Share within Each Segment (%)",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Investment Patterns by Cluster")

    col1, col2 = st.columns(2)
    with col1:
        spend_by_seg = filtered.groupby("segment")["total_spend"].mean().reset_index()
        fig6 = px.bar(
            spend_by_seg, x="segment", y="total_spend", color="segment",
            color_discrete_map=SEGMENT_COLORS, title="Average Total Spend per Client by Segment",
            text_auto=".2s",
        )
        fig6.update_layout(showlegend=False, yaxis_title="Avg. Spend ($)")
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        price_by_seg = filtered.groupby("segment")["avg_unit_price"].mean().reset_index()
        fig7 = px.bar(
            price_by_seg, x="segment", y="avg_unit_price", color="segment",
            color_discrete_map=SEGMENT_COLORS, title="Average Unit Price by Segment",
            text_auto=".2s",
        )
        fig7.update_layout(showlegend=False, yaxis_title="Avg. Unit Price ($)")
        st.plotly_chart(fig7, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        units_by_seg = filtered.groupby("segment")["num_units_purchased"].mean().reset_index()
        fig8 = px.bar(
            units_by_seg, x="segment", y="num_units_purchased", color="segment",
            color_discrete_map=SEGMENT_COLORS, title="Average Units Purchased by Segment",
        )
        fig8.update_layout(showlegend=False, yaxis_title="Avg. Units")
        st.plotly_chart(fig8, use_container_width=True)

    with col4:
        loan_by_seg = (
            filtered.groupby("segment")["loan_applied"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="pct_loan")
        )
        fig9 = px.bar(
            loan_by_seg, x="segment", y="pct_loan", color="segment",
            color_discrete_map=SEGMENT_COLORS, title="% of Clients Using Financing (Loan Applied)",
        )
        fig9.update_layout(showlegend=False, yaxis_title="% Loan Applied")
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Spend vs. Unit Price (bubble = units purchased)")
    fig10 = px.scatter(
        filtered, x="avg_unit_price", y="total_spend", size="num_units_purchased",
        color="segment", color_discrete_map=SEGMENT_COLORS, hover_data=["client_id", "country"],
        title="Client Investment Profile", opacity=0.7,
    )
    st.plotly_chart(fig10, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: GEOGRAPHIC BUYER ANALYSIS
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Buyer Segments by Geography")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        geo = filtered.groupby(["country", "segment"]).size().reset_index(name="count")
        fig11 = px.bar(
            geo, x="country", y="count", color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Buyer Segment Distribution by Country", barmode="stack",
        )
        fig11.update_layout(xaxis_title="", yaxis_title="Clients")
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        country_totals = filtered.groupby("country")["total_spend"].sum().reset_index()
        fig12 = px.choropleth(
            country_totals, locations="country", locationmode="country names",
            color="total_spend", color_continuous_scale="Blues",
            title="Total Investment Value by Country",
        )
        st.plotly_chart(fig12, use_container_width=True)

    st.subheader("Top Regions by Client Count")
    top_regions = filtered["region"].value_counts().head(15).reset_index()
    top_regions.columns = ["region", "count"]
    fig13 = px.bar(top_regions.sort_values("count"), x="count", y="region", orientation="h",
                    title="Top 15 Regions by Number of Clients")
    fig13.update_layout(yaxis_title="", xaxis_title="Clients")
    st.plotly_chart(fig13, use_container_width=True)

    st.subheader("Average Spend by Country")
    avg_spend_country = filtered.groupby("country")["total_spend"].mean().reset_index().sort_values("total_spend")
    fig14 = px.bar(avg_spend_country, x="total_spend", y="country", orientation="h",
                    title="Average Client Spend by Country", text_auto=".2s")
    fig14.update_layout(yaxis_title="", xaxis_title="Avg. Spend ($)")
    st.plotly_chart(fig14, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: SEGMENT INSIGHTS PANEL
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Descriptive Statistics per Cluster")

    for seg in sorted(filtered["segment"].unique(), key=lambda s: list(SEGMENT_COLORS).index(s) if s in SEGMENT_COLORS else 99):
        seg_df = filtered[filtered["segment"] == seg]
        with st.container(border=True):
            st.markdown(f"### {seg}  \n*{SEGMENT_DESCRIPTIONS.get(seg, '')}*")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Clients", f"{len(seg_df):,}")
            c2.metric("Avg. Age", f"{seg_df['age'].mean():.1f}")
            c3.metric("Avg. Spend", f"${seg_df['total_spend'].mean():,.0f}")
            c4.metric("Avg. Unit Price", f"${seg_df['avg_unit_price'].mean():,.0f}")
            c5.metric("Avg. Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f} / 5")

    st.markdown("---")
    st.subheader("Full Cluster Profile Table")
    profile_display = filtered.groupby("segment").agg(
        clients=("client_id", "count"),
        avg_age=("age", "mean"),
        pct_investment_purpose=("acquisition_purpose", lambda x: (x == "Investment").mean() * 100),
        pct_loan_applied=("loan_applied", lambda x: (x == "Yes").mean() * 100),
        pct_company=("client_type", lambda x: (x == "Company").mean() * 100),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_units_purchased=("num_units_purchased", "mean"),
        avg_towers=("num_towers", "mean"),
        avg_spend=("total_spend", "mean"),
        avg_unit_price=("avg_unit_price", "mean"),
    ).round(2)
    st.dataframe(profile_display, use_container_width=True)

    st.markdown("---")
    st.subheader("Explore Raw Client-Level Data")
    st.dataframe(
        filtered[[
            "client_id", "client_type", "gender", "country", "region", "age",
            "acquisition_purpose", "loan_applied", "referral_channel", "satisfaction_score",
            "num_units_purchased", "total_spend", "avg_unit_price", "segment",
        ]],
        use_container_width=True, height=350,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "filtered_buyer_segments.csv", "text/csv")

st.markdown("---")
st.caption("Machine Learning based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence — "
            "Unified Mentor x Parcl Co. Limited")
