import gc

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Financial Fraud Detection Dashboard")
st.markdown(
    "An interactive analysis of financial transactions "
    "using the PaySim dataset."
)

st.caption(
    "Explore transaction patterns, identify suspicious activity, "
    "and compare fraudulent and legitimate transactions."
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "paysim.csv"
GOOGLE_DRIVE_FILE_ID = "1JZ0l24jtXO5CkjvRZzaeRnCkmP_7ECS3"


def download_dataset():
    """Download the large CSV from Google Drive using gdown (streams to disk)."""
    import gdown

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PATH.with_suffix(".tmp")

    result = gdown.download(
        id=GOOGLE_DRIVE_FILE_ID,
        output=str(tmp_path),
        quiet=False,
    )

    if result is None or not tmp_path.exists() or tmp_path.stat().st_size < 1_000_000:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Download failed. Make sure the file is shared as "
            "'Anyone with the link'."
        )

    # Rename only after the download fully completed
    tmp_path.rename(DATA_PATH)


TYPE_DTYPE = pd.CategoricalDtype(
    categories=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)


@st.cache_resource(show_spinner="Loading transaction dataset (first load takes about a minute)...")
def load_data(file_path):
    """Read the big CSV in chunks with compact dtypes to keep RAM low."""
    dtypes = {
        "step": "int16",
        "type": TYPE_DTYPE,
        "amount": "float64",
        "oldbalanceOrg": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }

    parts = []
    for chunk in pd.read_csv(
        file_path,
        usecols=list(dtypes),
        dtype=dtypes,
        chunksize=500_000,
    ):
        parts.append(chunk)

    data = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return data


if not DATA_PATH.exists():
    with st.spinner("Downloading dataset from Google Drive (~500 MB)..."):
        try:
            download_dataset()
        except Exception as e:
            st.error(f"Dataset download failed: {e}")
            st.stop()

df = load_data(str(DATA_PATH))

# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

transaction_types = sorted(df["type"].dropna().astype(str).unique().tolist())

selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=transaction_types,
    default=transaction_types,
)

fraud_options = {
    "All Transactions": None,
    "Legitimate Only": 0,
    "Fraud Only": 1,
}

selected_status = st.sidebar.selectbox(
    "Transaction Status",
    options=list(fraud_options.keys()),
)

max_amount = int(np.ceil(df["amount"].max()))

selected_amount = st.sidebar.slider(
    "Transaction Amount Range",
    min_value=0,
    max_value=max_amount,
    value=(0, max_amount),
    step=max(1, max_amount // 1000),
    format="%d",
)

# --------------------------------------------------
# 4. APPLY FILTERS
# --------------------------------------------------
filtered = df[
    df["type"].isin(selected_types)
    & df["amount"].between(selected_amount[0], selected_amount[1])
]

fraud_value = fraud_options[selected_status]

if fraud_value is not None:
    filtered = filtered[filtered["isFraud"] == fraud_value]

st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} records")

if filtered.empty:
    st.warning(
        "No transactions match these filters. "
        "Please change your selections."
    )
    st.stop()

# --------------------------------------------------
# 5. KPI CARDS
# --------------------------------------------------
total_records = len(filtered)

fraud_mask = filtered["isFraud"] == 1
fraud_data = filtered[fraud_mask]

total_frauds = len(fraud_data)
fraud_rate = total_frauds / total_records * 100 if total_records else 0
average_amount = filtered["amount"].mean()
total_fraud_amount = fraud_data["amount"].sum()

st.subheader("Key Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Transactions", f"{total_records:,}")
k2.metric("Fraud Transactions", f"{total_frauds:,}")
k3.metric("Fraud Rate", f"{fraud_rate:.4f}%")
k4.metric("Average Transaction", f"{average_amount:,.2f}")
k5.metric("Total Fraud Amount", f"{total_fraud_amount:,.2f}")

st.divider()

# --------------------------------------------------
# 6. PREPARE CHART DATA (memory-friendly, no full copy)
# --------------------------------------------------
status_counts = (
    filtered["isFraud"]
    .map({0: "Legitimate", 1: "Fraud"})
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Transactions")
)

type_counts = (
    filtered.groupby("type", observed=True)
    .size()
    .reset_index(name="Transactions")
)

# Sample: keep ALL fraud rows (very few) + a sample of legitimate rows,
# so fraud is properly represented in the detailed charts.
legit_positions = np.flatnonzero(~fraud_mask.to_numpy())
rng = np.random.default_rng(42)
if len(legit_positions) > 50000:
    legit_positions = rng.choice(legit_positions, size=50000, replace=False)
legit_part = filtered.iloc[legit_positions]

chart_sample = pd.concat([fraud_data, legit_part])
chart_sample["Fraud Status"] = chart_sample["isFraud"].map(
    {0: "Legitimate", 1: "Fraud"}
)

# --------------------------------------------------
# 7. VISUALIZATION 1: FRAUD DISTRIBUTION
# --------------------------------------------------
st.subheader("1. Fraud vs Legitimate Transactions")

fig1 = px.pie(
    status_counts,
    names="Status",
    values="Transactions",
    hole=0.45,
    title="Distribution of Transaction Status",
)

st.plotly_chart(fig1, width="stretch")

# --------------------------------------------------
# 8. VISUALIZATION 2: TRANSACTION TYPES
# --------------------------------------------------
st.subheader("2. Transactions by Type")

fig2 = px.bar(
    type_counts,
    x="type",
    y="Transactions",
    color="type",
    title="Number of Transactions for Each Type",
    labels={
        "type": "Transaction Type",
        "Transactions": "Number of Transactions",
    },
)

st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------
# 9. VISUALIZATION 3: FRAUD BY TRANSACTION TYPE
# --------------------------------------------------
st.subheader("3. Fraud Transactions by Type")

fraud_by_type = (
    fraud_data.groupby("type", observed=True)
    .size()
    .reset_index(name="Fraud Transactions")
)

fig3 = px.bar(
    fraud_by_type,
    x="type",
    y="Fraud Transactions",
    color="type",
    title="Fraud Count Across Transaction Types",
    labels={
        "type": "Transaction Type",
        "Fraud Transactions": "Fraud Count",
    },
)

st.plotly_chart(fig3, width="stretch")

# --------------------------------------------------
# 10. VISUALIZATION 4: AMOUNT DISTRIBUTION
# --------------------------------------------------
st.subheader("4. Transaction Amount Distribution")

fig4 = px.histogram(
    chart_sample,
    x="amount",
    color="Fraud Status",
    nbins=50,
    barmode="overlay",
    opacity=0.7,
    title="Transaction Amounts: Fraud vs Legitimate (sampled)",
    labels={
        "amount": "Transaction Amount",
        "count": "Number of Transactions",
    },
)

st.plotly_chart(fig4, width="stretch")

# --------------------------------------------------
# 11. VISUALIZATION 5: SENDER BALANCE COMPARISON
# --------------------------------------------------
st.subheader("5. Sender Balance Comparison")

fig5 = px.box(
    chart_sample,
    x="Fraud Status",
    y="oldbalanceOrg",
    color="Fraud Status",
    points=False,
    title="Original Sender Balance by Transaction Status (sampled)",
    labels={
        "Fraud Status": "Transaction Status",
        "oldbalanceOrg": "Original Sender Balance",
    },
)

st.plotly_chart(fig5, width="stretch")

# --------------------------------------------------
# 12. VISUALIZATION 6: FRAUD OVER TIME
# --------------------------------------------------
st.subheader("6. Fraud Transactions Over Time")

fraud_trend = (
    fraud_data.groupby("step")
    .size()
    .reset_index(name="Fraud Transactions")
    .sort_values("step")
)

fig6 = px.line(
    fraud_trend,
    x="step",
    y="Fraud Transactions",
    markers=True,
    title="Fraud Count Across Simulation Steps",
    labels={
        "step": "Simulation Step",
        "Fraud Transactions": "Fraud Count",
    },
)

st.plotly_chart(fig6, width="stretch")

# --------------------------------------------------
# 13. KEY INSIGHTS AND FINDINGS
# --------------------------------------------------
st.divider()
st.header("Key Insights and Findings")

st.caption(
    "These findings are calculated from the transactions "
    "currently selected in the dashboard filters."
)

most_common_type = filtered["type"].value_counts().idxmax()

if total_frauds > 0:
    most_fraudulent_type = fraud_data["type"].value_counts().idxmax()
    largest_fraud = fraud_data["amount"].max()
else:
    most_fraudulent_type = "No fraud records in selection"
    largest_fraud = 0

legit_amounts = filtered.loc[~fraud_mask, "amount"]

if len(legit_amounts) > 0 and total_frauds > 0:
    fraud_median = fraud_data["amount"].median()
    legitimate_median = legit_amounts.median()

    if legitimate_median > 0:
        median_comparison = (
            f"The median fraud transaction amount is "
            f"{fraud_median / legitimate_median:.2f} times "
            f"the median legitimate transaction amount."
        )
    else:
        median_comparison = (
            f"Median fraud amount: {fraud_median:,.2f}; "
            f"median legitimate amount: {legitimate_median:,.2f}."
        )
else:
    median_comparison = (
        "Select both fraud and legitimate transactions "
        "to compare their median amounts."
    )

flagged_frauds = int(fraud_data["isFlaggedFraud"].sum())

insights = [
    (
        "Transaction volume",
        f"{total_records:,} transactions match the current filters.",
    ),
    (
        "Fraud prevalence",
        f"{total_frauds:,} transactions are fraudulent, "
        f"representing {fraud_rate:.4f}% of the selected records.",
    ),
    (
        "Most common transaction type",
        f"{most_common_type} has the highest transaction count "
        "in the selected data.",
    ),
    (
        "Fraud-prone transaction type",
        f"{most_fraudulent_type} has the highest fraud count "
        "among the selected transaction types.",
    ),
    ("Amount comparison", median_comparison),
    (
        "Flagged fraud records",
        f"{flagged_frauds:,} fraud records in the selection "
        "have isFlaggedFraud = 1.",
    ),
    (
        "Largest fraud transaction",
        f"The largest selected fraudulent transaction amount "
        f"is {largest_fraud:,.2f}.",
    ),
]

for heading, detail in insights:
    st.markdown(f"**{heading}:** {detail}")

# --------------------------------------------------
# 14. DATA PREVIEW AND DOWNLOAD
# --------------------------------------------------
with st.expander("View filtered transaction data"):
    st.dataframe(filtered.head(100), width="stretch")

    st.caption("The preview shows at most 100 rows for performance.")

    csv_data = filtered.head(100_000).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download first 100,000 filtered rows (CSV)",
        data=csv_data,
        file_name="filtered_financial_transactions.csv",
        mime="text/csv",
    )

st.caption(
    "Project: Financial Fraud Detection | "
    "Dataset: PaySim | Built with Streamlit, Pandas and Plotly"
)
