
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

@st.cache_data(show_spinner="Loading transaction dataset...")
def load_data(file_path):
    columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud"
    ]

    data = pd.read_csv(
        file_path,
        usecols=columns,
        dtype={
            "step": "int32",
            "type": "category",
            "amount": "float64",
            "oldbalanceOrg": "float64",
            "newbalanceOrig": "float64",
            "oldbalanceDest": "float64",
            "newbalanceDest": "float64",
            "isFraud": "int8",
            "isFlaggedFraud": "int8"
        }
    )

    return data


import gdown

if not DATA_PATH.exists():
    with st.spinner("Downloading dataset from Google Drive..."):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

                downloaded_file = gdown.download(
            url,
            str(DATA_PATH),
            quiet=False
        )

        if downloaded_file is None or not DATA_PATH.exists():
            st.error("Dataset download failed. Please check the Google Drive link.")
            st.stop()

df = load_data(str(DATA_PATH))

# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

transaction_types = sorted(
    df["type"].dropna().astype(str).unique().tolist()
)

selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=transaction_types,
    default=transaction_types
)

fraud_options = {
    "All Transactions": None,
    "Legitimate Only": 0,
    "Fraud Only": 1
}

selected_status = st.sidebar.selectbox(
    "Transaction Status",
    options=list(fraud_options.keys())
)

max_amount = int(df["amount"].max())

selected_amount = st.sidebar.slider(
    "Transaction Amount Range",
    min_value=0,
    max_value=max_amount,
    value=(0, max_amount),
    step=max(1, max_amount // 1000),
    format="%d"
)

# --------------------------------------------------
# 4. APPLY FILTERS
# --------------------------------------------------
filtered = df[
    df["type"].astype(str).isin(selected_types)
    & df["amount"].between(
        selected_amount[0], selected_amount[1]
    )
]

fraud_value = fraud_options[selected_status]

if fraud_value is not None:
    filtered = filtered[
        filtered["isFraud"] == fraud_value
    ]

st.sidebar.caption(
    f"Showing {len(filtered):,} of {len(df):,} records"
)

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

fraud_data = filtered[filtered["isFraud"] == 1]
legitimate_data = filtered[filtered["isFraud"] == 0]

total_frauds = len(fraud_data)

fraud_rate = (
    total_frauds / total_records * 100
    if total_records else 0
)

average_amount = filtered["amount"].mean()

total_fraud_amount = fraud_data["amount"].sum()

st.subheader("Key Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Transactions",
    f"{total_records:,}"
)

k2.metric(
    "Fraud Transactions",
    f"{total_frauds:,}"
)

k3.metric(
    "Fraud Rate",
    f"{fraud_rate:.4f}%"
)

k4.metric(
    "Average Transaction",
    f"{average_amount:,.2f}"
)

k5.metric(
    "Total Fraud Amount",
    f"{total_fraud_amount:,.2f}"
)

st.divider()

# --------------------------------------------------
# 6. PREPARE CHART DATA
# --------------------------------------------------
chart_data = filtered.copy()

chart_data["Fraud Status"] = chart_data["isFraud"].map(
    {0: "Legitimate", 1: "Fraud"}
)

# Sampling keeps detailed charts responsive on large data.
chart_sample = chart_data.sample(
    n=min(50000, len(chart_data)),
    random_state=42
)

# --------------------------------------------------
# 7. VISUALIZATION 1: FRAUD DISTRIBUTION
# --------------------------------------------------
st.subheader("1. Fraud vs Legitimate Transactions")

status_counts = (
    chart_data["Fraud Status"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Transactions")
)

fig1 = px.pie(
    status_counts,
    names="Status",
    values="Transactions",
    hole=0.45,
    title="Distribution of Transaction Status"
)

st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------
# 8. VISUALIZATION 2: TRANSACTION TYPES
# --------------------------------------------------
st.subheader("2. Transactions by Type")

type_counts = (
    chart_data.groupby("type", observed=True)
    .size()
    .reset_index(name="Transactions")
)

fig2 = px.bar(
    type_counts,
    x="type",
    y="Transactions",
    color="type",
    title="Number of Transactions for Each Type",
    labels={
        "type": "Transaction Type",
        "Transactions": "Number of Transactions"
    }
)

st.plotly_chart(fig2, use_container_width=True)

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
        "Fraud Transactions": "Fraud Count"
    }
)

st.plotly_chart(fig3, use_container_width=True)

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
    title="Transaction Amounts: Fraud vs Legitimate",
    labels={
        "amount": "Transaction Amount",
        "count": "Number of Transactions"
    }
)

st.plotly_chart(fig4, use_container_width=True)

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
    title="Original Sender Balance by Transaction Status",
    labels={
        "Fraud Status": "Transaction Status",
        "oldbalanceOrg": "Original Sender Balance"
    }
)

st.plotly_chart(fig5, use_container_width=True)

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
        "Fraud Transactions": "Fraud Count"
    }
)

st.plotly_chart(fig6, use_container_width=True)

# --------------------------------------------------
# 13. KEY INSIGHTS AND FINDINGS
# --------------------------------------------------
st.divider()
st.header("Key Insights and Findings")

st.caption(
    "These findings are calculated from the transactions "
    "currently selected in the dashboard filters."
)

most_common_type = (
    filtered["type"].value_counts().idxmax()
    if not filtered.empty else "N/A"
)

if total_frauds > 0:
    most_fraudulent_type = (
        fraud_data["type"].value_counts().idxmax()
    )
    largest_fraud = fraud_data["amount"].max()
else:
    most_fraudulent_type = "No fraud records in selection"
    largest_fraud = 0

if not legitimate_data.empty and not fraud_data.empty:
    fraud_median = fraud_data["amount"].median()
    legitimate_median = legitimate_data["amount"].median()

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

flagged_frauds = int(
    filtered.loc[
        filtered["isFraud"] == 1, "isFlaggedFraud"
    ].sum()
)

insights = [
    (
        "Transaction volume",
        f"{total_records:,} transactions match the current filters."
    ),
    (
        "Fraud prevalence",
        f"{total_frauds:,} transactions are fraudulent, "
        f"representing {fraud_rate:.4f}% of the selected records."
    ),
    (
        "Most common transaction type",
        f"{most_common_type} has the highest transaction count "
        "in the selected data."
    ),
    (
        "Fraud-prone transaction type",
        f"{most_fraudulent_type} has the highest fraud count "
        "among the selected transaction types."
    ),
    (
        "Amount comparison",
        median_comparison
    ),
    (
        "Flagged fraud records",
        f"{flagged_frauds:,} fraud records in the selection "
        "have isFlaggedFraud = 1."
    ),
    (
        "Largest fraud transaction",
        f"The largest selected fraudulent transaction amount "
        f"is {largest_fraud:,.2f}."
    )
]

for heading, detail in insights:
    st.markdown(f"**{heading}:** {detail}")

# --------------------------------------------------
# 14. DATA PREVIEW AND DOWNLOAD
# --------------------------------------------------
with st.expander("View filtered transaction data"):
    st.dataframe(
        filtered.head(100),
        use_container_width=True
    )

    st.caption(
        "The preview shows at most 100 rows for performance."
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_financial_transactions.csv",
        mime="text/csv"
    )

st.caption(
    "Project: Financial Fraud Detection | "
    "Dataset: PaySim | Built with Streamlit, Pandas and Plotly"
)
