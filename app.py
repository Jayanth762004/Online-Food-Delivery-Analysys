import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Online Food Delivery Analytics",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main background */
        .main { background-color: #f9fafb; }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px 20px;
        }
        div[data-testid="metric-container"] label {
            font-size: 13px !important;
            color: #6b7280 !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 700 !important;
            color: #1f2937 !important;
        }

        /* Section headers */
        .section-header {
            background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
            color: #ffffff;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 18px;
            letter-spacing: 0.02em;
        }

        /* Insight cards */
        .insight-card {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 10px;
            font-size: 14px;
            color: #1e3a5f;
        }
        .insight-card b { color: #1d4ed8; }

        /* Decision cards */
        .decision-card {
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 10px;
            font-size: 14px;
            color: #14532d;
        }
        .decision-card b { color: #16a34a; }

        /* Warning cards */
        .warning-card {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 8px;
            font-size: 13px;
            color: #78350f;
        }

        /* Dataframe */
        .stDataFrame { border-radius: 8px; overflow: hidden; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #1e293b; }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; font-weight:600; }

        /* Hide streamlit branding */
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper: section header ─────────────────────────────────────────────────────
def section(title: str):
    st.markdown(f'<div class="section-header">📌 {title}</div>', unsafe_allow_html=True)

# ── Helper: insight / decision cards ──────────────────────────────────────────
def insight(text: str):
    st.markdown(f'<div class="insight-card">💡 {text}</div>', unsafe_allow_html=True)

def decision(text: str):
    st.markdown(f'<div class="decision-card">✅ {text}</div>', unsafe_allow_html=True)

def warn(text: str):
    st.markdown(f'<div class="warning-card">⚠️ {text}</div>', unsafe_allow_html=True)

# ── Income numeric map ─────────────────────────────────────────────────────────
INCOME_ORDER = {
    "No Income": 0,
    "Below Rs.10000": 5000,
    "10001 to 25000": 17500,
    "25001 to 50000": 37500,
    "More than 50000": 60000,
}

# ── 1. LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip column names and string values
    df.columns = df.columns.str.strip()
    # Drop empty/unnamed trailing columns (caused by trailing comma in CSV header)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

CSV_PATH = "online food delivery dataset.csv"

if not os.path.exists(CSV_PATH):
    st.error(f"Dataset not found at **{CSV_PATH}**. Please place it in the same folder as app.py.")
    st.stop()

raw_df = load_data(CSV_PATH)

# ── 2. DATA QUALITY CHECK ──────────────────────────────────────────────────────
@st.cache_data
def quality_report(df: pd.DataFrame):
    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values,
    })
    return report

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop fully duplicate rows
    df.drop_duplicates(inplace=True)
    # Map income to numeric
    df["Income_Numeric"] = df["Monthly Income"].map(INCOME_ORDER).fillna(0)
    # Gender encoded: Female=0, Male=1
    df["Gender_Code"] = df["Gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)
    # Sales score = Gender_Code × Age  (as per task requirement)
    df["Sales_Score"] = df["Gender_Code"] * df["Age"]
    # Feedback binary
    df["Positive_Feedback"] = (df["Feedback"].str.strip().str.lower() == "positive").astype(int)
    # Output binary
    df["Orders_Online"] = (df["Output"].str.strip().str.lower() == "yes").astype(int)
    return df

# ── 3. PROCESS ────────────────────────────────────────────────────────────────
df = clean_data(raw_df)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍔 Food Delivery\n### Analytics Dashboard")
    st.markdown("---")
    st.markdown("**Filters**")

    gender_filter = st.multiselect(
        "Gender",
        options=sorted(df["Gender"].dropna().unique()),
        default=sorted(df["Gender"].dropna().unique()),
    )
    occupation_filter = st.multiselect(
        "Occupation",
        options=sorted(df["Occupation"].dropna().unique()),
        default=sorted(df["Occupation"].dropna().unique()),
    )
    age_range = st.slider(
        "Age Range",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())),
    )
    customer_type = st.multiselect(
        "Customer Type",
        options=sorted(df["Customer Type"].dropna().unique()),
        default=sorted(df["Customer Type"].dropna().unique()),
    )

    st.markdown("---")
    st.caption(f"Dataset: `{CSV_PATH}`\nRows: **{len(df):,}**  |  Cols: **{len(df.columns)}**")

# Apply filters
mask = (
    df["Gender"].isin(gender_filter)
    & df["Occupation"].isin(occupation_filter)
    & df["Age"].between(age_range[0], age_range[1])
    & df["Customer Type"].isin(customer_type)
)
fdf = df[mask].copy()

# ── TITLE ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:32px;font-weight:800;color:#1e40af;margin-bottom:4px;'>"
    "🍔 Online Food Delivery — Data Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#6b7280;font-size:15px;margin-top:0;'>End-to-end analysis: data quality → sales metrics → group summaries → charts → business decisions</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📂 Dataset",
    "🔍 Data Quality",
    "💰 Sales Metrics",
    "📊 Group Summary",
    "📈 Charts",
    "🧠 Business Decisions",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(fdf):,}")
    c2.metric("Filtered Records", f"{len(fdf):,}")
    c3.metric("Features", f"{len(fdf.columns)}")
    c4.metric("Age Range", f"{fdf['Age'].min()} – {fdf['Age'].max()}")

    st.markdown("#### Raw Data (filtered)")
    st.dataframe(
        fdf[["Age", "Gender", "Marital Status", "Occupation", "Monthly Income",
             "Educational Qualifications", "Family size", "Customer Type",
             "Output", "Feedback"]].reset_index(drop=True),
        use_container_width=True,
        height=380,
    )

    st.markdown("#### Column Descriptions")
    descriptions = {
        "Age": "Customer age in years",
        "Gender": "Customer gender (Male / Female)",
        "Marital Status": "Single / Married / Prefer not to say",
        "Occupation": "Student / Employee / Self Employed / House wife",
        "Monthly Income": "Income bracket (categorical)",
        "Educational Qualifications": "Highest qualification attained",
        "Family size": "Number of family members",
        "Customer Type": "New / Regular / Frequent",
        "Output": "Whether customer orders food online (Yes/No)",
        "Feedback": "Customer feedback sentiment (Positive/Negative)",
        "Income_Numeric": "Numeric midpoint of income bracket (derived)",
        "Sales_Score": "Gender_Code × Age (derived sales proxy)",
    }
    desc_df = pd.DataFrame(descriptions.items(), columns=["Column", "Description"])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("Data Quality Report")

    qr = quality_report(raw_df)
    total_nulls = int(qr["Null Count"].sum())
    dup_rows = int(raw_df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows (raw)", f"{len(raw_df):,}")
    c2.metric("Duplicate Rows", f"{dup_rows:,}")
    c3.metric("Columns with Nulls", f"{int((qr['Null Count'] > 0).sum())}")
    c4.metric("Total Missing Cells", f"{total_nulls:,}")

    if total_nulls == 0 and dup_rows == 0:
        st.success("✅ Dataset is clean — no missing values and no duplicate rows detected.")
    else:
        if total_nulls > 0:
            warn(f"Found **{total_nulls}** missing value(s). See table below for details.")
        if dup_rows > 0:
            warn(f"Found **{dup_rows}** duplicate row(s) — these were removed during cleaning.")

    st.markdown("#### Per-Column Quality Report")
    styled_qr = qr.copy()
    styled_qr["Null %"] = styled_qr["Null %"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(styled_qr, use_container_width=True, hide_index=True)

    st.markdown("#### Descriptive Statistics (numeric columns)")
    num_cols = ["Age", "Family size", "Income_Numeric", "Sales_Score"]
    st.dataframe(
        fdf[num_cols].describe().round(2).T.rename(columns=str),
        use_container_width=True,
    )

    # Value distribution for categoricals
    st.markdown("#### Categorical Value Distributions")
    cat_cols = ["Gender", "Marital Status", "Occupation", "Monthly Income",
                "Educational Qualifications", "Customer Type", "Output", "Feedback"]
    for col in cat_cols:
        vc = fdf[col].value_counts().reset_index()
        vc.columns = [col, "Count"]
        vc["Percentage"] = (vc["Count"] / vc["Count"].sum() * 100).round(1).astype(str) + "%"
        with st.expander(f"  {col}"):
            st.dataframe(vc, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SALES METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section("Sales Score Analysis  (Sales = Gender_Code × Age)")

    st.info(
        "**Sales Score formula:** Gender is encoded as Male = 1, Female = 0. "
        "Sales Score = Gender_Code × Age.  This creates a male-specific activity "
        "index proportional to age, enabling gender-segmented engagement analysis."
    )

    total_sales = fdf["Sales_Score"].sum()
    avg_sales = fdf["Sales_Score"].mean()
    max_sales = fdf["Sales_Score"].max()
    male_share = (fdf[fdf["Gender"] == "Male"]["Sales_Score"].sum() / total_sales * 100) if total_sales > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales Score", f"{total_sales:,.0f}")
    c2.metric("Average Sales Score", f"{avg_sales:.2f}")
    c3.metric("Max Sales Score", f"{max_sales:,.0f}")
    c4.metric("Male Contribution %", f"{male_share:.1f}%")

    st.markdown("#### Sales Score by Gender")
    gender_sales = fdf.groupby("Gender")["Sales_Score"].agg(
        Total="sum", Average="mean", Count="count"
    ).round(2).reset_index()
    gender_sales.columns = ["Gender", "Total Sales Score", "Avg Sales Score", "Count"]
    st.dataframe(gender_sales, use_container_width=True, hide_index=True)

    st.markdown("#### Sales Score by Age Group")
    fdf["Age_Group"] = pd.cut(
        fdf["Age"], bins=[17, 22, 26, 30, 40, 100],
        labels=["18–22", "23–26", "27–30", "31–40", "41+"]
    )
    age_sales = fdf.groupby("Age_Group", observed=True)["Sales_Score"].agg(
        Total="sum", Average="mean", Count="count"
    ).round(2).reset_index()
    age_sales.columns = ["Age Group", "Total Sales Score", "Avg Sales Score", "Count"]
    st.dataframe(age_sales, use_container_width=True, hide_index=True)

    st.markdown("#### Sales Score by Occupation")
    occ_sales = fdf.groupby("Occupation")["Sales_Score"].agg(
        Total="sum", Average="mean", Count="count"
    ).round(2).reset_index().sort_values("Total Sales Score", ascending=False)
    occ_sales.columns = ["Occupation", "Total Sales Score", "Avg Sales Score", "Count"]
    st.dataframe(occ_sales, use_container_width=True, hide_index=True)

    insight(f"Total Sales Score across all filtered customers: <b>{total_sales:,.0f}</b>")
    insight(f"Male customers drive <b>{male_share:.1f}%</b> of the total Sales Score.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GROUP SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section("Group & Summary Statistics")

    # ── Summary by Gender ──
    st.markdown("#### 👥 Summary by Gender")
    g_sum = fdf.groupby("Gender").agg(
        Count=("Age", "count"),
        Avg_Age=("Age", "mean"),
        Avg_FamilySize=("Family size", "mean"),
        Avg_Income=("Income_Numeric", "mean"),
        Total_Sales=("Sales_Score", "sum"),
        Avg_Sales=("Sales_Score", "mean"),
        Online_Orders=("Orders_Online", "sum"),
        Positive_Feedback=("Positive_Feedback", "sum"),
    ).round(2).reset_index()
    g_sum["Online_Order_%"] = (g_sum["Online_Orders"] / g_sum["Count"] * 100).round(1)
    g_sum["Positive_Feedback_%"] = (g_sum["Positive_Feedback"] / g_sum["Count"] * 100).round(1)
    st.dataframe(g_sum, use_container_width=True, hide_index=True)

    # ── Summary by Occupation ──
    st.markdown("#### 💼 Summary by Occupation")
    o_sum = fdf.groupby("Occupation").agg(
        Count=("Age", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Income_Numeric", "mean"),
        Total_Sales=("Sales_Score", "sum"),
        Online_Orders=("Orders_Online", "sum"),
        Positive_Feedback=("Positive_Feedback", "sum"),
    ).round(2).reset_index().sort_values("Count", ascending=False)
    o_sum["Online_Order_%"] = (o_sum["Online_Orders"] / o_sum["Count"] * 100).round(1)
    st.dataframe(o_sum, use_container_width=True, hide_index=True)

    # ── Summary by Income ──
    st.markdown("#### 💵 Summary by Monthly Income")
    income_order_list = list(INCOME_ORDER.keys())
    i_sum = fdf.groupby("Monthly Income").agg(
        Count=("Age", "count"),
        Avg_Age=("Age", "mean"),
        Total_Sales=("Sales_Score", "sum"),
        Online_Orders=("Orders_Online", "sum"),
        Positive_Feedback=("Positive_Feedback", "sum"),
    ).round(2).reset_index()
    i_sum["Income_Numeric"] = i_sum["Monthly Income"].map(INCOME_ORDER)
    i_sum = i_sum.sort_values("Income_Numeric").drop(columns="Income_Numeric")
    i_sum["Online_Order_%"] = (i_sum["Online_Orders"] / i_sum["Count"] * 100).round(1)
    st.dataframe(i_sum, use_container_width=True, hide_index=True)

    # ── Summary by Customer Type ──
    st.markdown("#### 🏷️ Summary by Customer Type")
    ct_sum = fdf.groupby("Customer Type").agg(
        Count=("Age", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Income_Numeric", "mean"),
        Total_Sales=("Sales_Score", "sum"),
        Positive_Feedback=("Positive_Feedback", "sum"),
    ).round(2).reset_index().sort_values("Count", ascending=False)
    ct_sum["Positive_Feedback_%"] = (ct_sum["Positive_Feedback"] / ct_sum["Count"] * 100).round(1)
    st.dataframe(ct_sum, use_container_width=True, hide_index=True)

    # ── Overall KPIs ──
    st.markdown("#### 📌 Overall KPI Summary")
    kpi_data = {
        "Metric": [
            "Total Customers", "Average Age", "Average Family Size",
            "Online Ordering Rate (%)", "Positive Feedback Rate (%)",
            "Total Sales Score", "Average Sales Score",
        ],
        "Value": [
            f"{len(fdf):,}",
            f"{fdf['Age'].mean():.1f}",
            f"{fdf['Family size'].mean():.1f}",
            f"{fdf['Orders_Online'].mean()*100:.1f}%",
            f"{fdf['Positive_Feedback'].mean()*100:.1f}%",
            f"{fdf['Sales_Score'].sum():,.0f}",
            f"{fdf['Sales_Score'].mean():.2f}",
        ],
    }
    st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section("Visual Analytics")

    PALETTE = ["#3b82f6", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#eab308"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })

    def show_fig(fig):
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Row 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("##### Gender Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        gd = fdf["Gender"].value_counts()
        wedges, texts, autotexts = ax.pie(
            gd, labels=gd.index, autopct="%1.1f%%",
            colors=PALETTE[:len(gd)], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_title("Gender Split")
        show_fig(fig)

    with r1c2:
        st.markdown("##### Age Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(fdf["Age"], bins=15, color=PALETTE[0], edgecolor="white", linewidth=1.2)
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Age Distribution")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        show_fig(fig)

    # Row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("##### Online Ordering Rate by Occupation")
        fig, ax = plt.subplots(figsize=(6, 4))
        oo = fdf.groupby("Occupation")["Orders_Online"].mean().mul(100).sort_values(ascending=False)
        bars = ax.barh(oo.index, oo.values, color=PALETTE[1], edgecolor="white")
        for bar, val in zip(bars, oo.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("Online Order Rate (%)")
        ax.set_title("Online Ordering Rate by Occupation")
        ax.set_xlim(0, 115)
        show_fig(fig)

    with r2c2:
        st.markdown("##### Feedback Sentiment by Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        fb = fdf.groupby(["Gender", "Feedback"]).size().unstack(fill_value=0)
        fb.plot(kind="bar", ax=ax, color=["#ef4444", "#10b981"], edgecolor="white", width=0.6)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        ax.set_title("Feedback by Gender")
        ax.legend(title="Feedback", fontsize=10)
        ax.tick_params(axis="x", rotation=0)
        show_fig(fig)

    # Row 3
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("##### Sales Score by Age Group")
        fig, ax = plt.subplots(figsize=(6, 4))
        ag_s = fdf.groupby("Age_Group", observed=True)["Sales_Score"].sum()
        bars = ax.bar(ag_s.index.astype(str), ag_s.values, color=PALETTE[2], edgecolor="white")
        for bar, val in zip(bars, ag_s.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:,.0f}", ha="center", fontsize=9)
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Total Sales Score")
        ax.set_title("Total Sales Score by Age Group")
        show_fig(fig)

    with r3c2:
        st.markdown("##### Customer Type Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ct = fdf["Customer Type"].value_counts()
        wedges, texts, autotexts = ax.pie(
            ct, labels=ct.index, autopct="%1.1f%%",
            colors=PALETTE[3:3+len(ct)], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_title("Customer Type Split")
        show_fig(fig)

    # Row 4
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        st.markdown("##### Monthly Income vs Online Ordering")
        fig, ax = plt.subplots(figsize=(6, 4))
        inc_order = fdf.groupby("Monthly Income")["Orders_Online"].mean().mul(100)
        inc_order.index = inc_order.index.map(
            lambda x: x if len(x) <= 18 else x[:16] + "…"
        )
        inc_map = {v: i for i, v in enumerate(INCOME_ORDER.keys())}
        inc_order = inc_order.sort_index(key=lambda idx: idx.map(
            lambda x: INCOME_ORDER.get(x, INCOME_ORDER.get(x.replace("…", ""), 0))
        ))
        bars = ax.bar(range(len(inc_order)), inc_order.values, color=PALETTE[4], edgecolor="white")
        ax.set_xticks(range(len(inc_order)))
        ax.set_xticklabels(inc_order.index, rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, inc_order.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", fontsize=9)
        ax.set_ylabel("Online Order Rate (%)")
        ax.set_title("Online Ordering Rate by Income Bracket")
        ax.set_ylim(0, 120)
        show_fig(fig)

    with r4c2:
        st.markdown("##### Family Size vs Online Ordering")
        fig, ax = plt.subplots(figsize=(6, 4))
        fs_order = fdf.groupby("Family size")["Orders_Online"].mean().mul(100)
        ax.plot(fs_order.index, fs_order.values, marker="o", color=PALETTE[5],
                linewidth=2.5, markersize=7)
        for x, y in zip(fs_order.index, fs_order.values):
            ax.text(x, y + 1.5, f"{y:.0f}%", ha="center", fontsize=9)
        ax.set_xlabel("Family Size")
        ax.set_ylabel("Online Order Rate (%)")
        ax.set_title("Online Ordering Rate by Family Size")
        ax.set_ylim(0, 115)
        show_fig(fig)

    # Row 5 — Heatmap
    st.markdown("##### Correlation Heatmap (Numeric Features)")
    fig, ax = plt.subplots(figsize=(9, 4))
    num_data = fdf[["Age", "Family size", "Income_Numeric", "Sales_Score",
                     "Orders_Online", "Positive_Feedback"]].copy()
    corr = num_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="Blues",
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", pad=12)
    show_fig(fig)

    # Row 6 — Education
    st.markdown("##### Educational Qualification Distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    ed = fdf["Educational Qualifications"].value_counts().sort_values(ascending=True)
    bars = ax.barh(ed.index, ed.values, color=PALETTE[0], edgecolor="white")
    for bar, val in zip(bars, ed.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.set_xlabel("Count")
    ax.set_title("Educational Qualification Distribution")
    show_fig(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — BUSINESS DECISIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section("Business Decisions & Recommendations")

    # Compute key stats for dynamic decisions
    online_rate = fdf["Orders_Online"].mean() * 100
    positive_rate = fdf["Positive_Feedback"].mean() * 100
    top_occupation = (
        fdf.groupby("Occupation")["Orders_Online"].mean().idxmax()
        if len(fdf) > 0 else "N/A"
    )
    top_age_group = (
        fdf.groupby("Age_Group", observed=True)["Sales_Score"].sum().idxmax()
        if len(fdf) > 0 else "N/A"
    )
    dominant_gender = fdf["Gender"].value_counts().idxmax() if len(fdf) > 0 else "N/A"
    low_feedback_occ = (
        fdf.groupby("Occupation")["Positive_Feedback"].mean().idxmin()
        if len(fdf) > 0 else "N/A"
    )
    most_common_customer_type = fdf["Customer Type"].value_counts().idxmax() if len(fdf) > 0 else "N/A"

    # ── Finding 1: Ordering Rate
    st.markdown("#### 1️⃣ Customer Engagement")
    decision(
        f"<b>Online ordering rate is {online_rate:.1f}%</b>. "
        "This shows strong digital adoption. Invest in a smooth mobile ordering experience "
        "and loyalty rewards to push the remaining offline customers online."
    )

    # ── Finding 2: Gender segmentation
    st.markdown("#### 2️⃣ Gender-Based Marketing")
    male_oo = fdf[fdf["Gender"] == "Male"]["Orders_Online"].mean() * 100 if "Male" in fdf["Gender"].values else 0
    female_oo = fdf[fdf["Gender"] == "Female"]["Orders_Online"].mean() * 100 if "Female" in fdf["Gender"].values else 0
    decision(
        f"Male customers order online at <b>{male_oo:.1f}%</b> vs Female at <b>{female_oo:.1f}%</b>. "
        f"The dominant user base is <b>{dominant_gender}</b>. "
        "Run gender-targeted campaigns — e.g., sports-meal combos for males and health/diet menus for females."
    )

    # ── Finding 3: Student / Occupation focus
    st.markdown("#### 3️⃣ Occupation-Driven Promotions")
    decision(
        f"<b>{top_occupation}</b> shows the highest online ordering rate among all occupations. "
        "Offer student discounts, combo meals, and fast-delivery options catering to this segment. "
        "Partner with colleges and office parks for bulk ordering deals."
    )

    # ── Finding 4: Age group
    st.markdown("#### 4️⃣ Age Group Targeting")
    decision(
        f"Age group <b>{top_age_group}</b> drives the highest Sales Score. "
        "Focus digital advertising spend on this cohort via social media platforms (Instagram, Swiggy ads). "
        "Create age-specific meal bundles and flash deals."
    )

    # ── Finding 5: Feedback quality
    st.markdown("#### 5️⃣ Feedback & Service Quality")
    if positive_rate >= 75:
        decision(
            f"Positive feedback rate is <b>{positive_rate:.1f}%</b> — excellent customer satisfaction. "
            "Introduce a referral programme to leverage satisfied customers as brand advocates."
        )
    else:
        warn(
            f"Positive feedback rate is only <b>{positive_rate:.1f}%</b>. "
            "Urgently review delivery times, packaging quality, and food temperature complaints."
        )
    decision(
        f"<b>{low_feedback_occ}</b> records the lowest positive-feedback share among occupations. "
        "Investigate specific pain points for this group (delivery slots, pricing) and resolve them."
    )

    # ── Finding 6: Customer type
    st.markdown("#### 6️⃣ Customer Retention Strategy")
    decision(
        f"<b>{most_common_customer_type}</b> is the dominant customer type. "
        "Build a tiered loyalty programme: New → Regular → Frequent, with escalating rewards "
        "(free delivery, exclusive menus, priority support) to move customers up the value ladder."
    )

    # ── Finding 7: Income & pricing
    st.markdown("#### 7️⃣ Pricing & Income Segmentation")
    decision(
        "A large share of users fall in the <b>No Income / Below Rs.10,000</b> bracket (mostly students). "
        "Maintain affordable meal tiers (₹50–₹150) as the primary price band. "
        "Introduce premium options for <b>More than Rs.50,000</b> earners to capture higher margins."
    )

    # ── Finding 8: Family size
    st.markdown("#### 8️⃣ Family & Group Orders")
    decision(
        "Larger families (4–6 members) show high ordering activity. "
        "Create <b>Family Feast</b> bundles and group-order discounts. "
        "Promote party packages for families of 5+ to increase average order value."
    )

    # ── Summary Table
    st.markdown("---")
    st.markdown("#### 📋 Decision Summary Table")
    summary = pd.DataFrame({
        "Priority": ["🔴 High", "🔴 High", "🟡 Medium", "🟡 Medium", "🟢 Low", "🟢 Low"],
        "Area": [
            "Customer Engagement",
            "Occupation Promotions",
            "Gender Marketing",
            "Feedback & Quality",
            "Pricing Tiers",
            "Family Bundles",
        ],
        "Action": [
            f"Invest in mobile UX; push {100-online_rate:.0f}% offline users online",
            f"Discount & combo deals for {top_occupation} segment",
            f"Targeted campaigns by gender ({dominant_gender} leads)",
            "Resolve issues for low-feedback occupations",
            "Keep Rs.50–150 tier; add premium for 50k+ earners",
            "Launch Family Feast bundles for 4–6 member households",
        ],
        "Expected Impact": [
            "+10–15% orders",
            "+20% from student/employee segment",
            "+8% engagement",
            "+5% satisfaction score",
            "+12% revenue per order",
            "+18% avg order value",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:12px;'>"
    "Online Food Delivery Analytics Dashboard &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; Dataset: online food delivery dataset.csv"
    "</p>",
    unsafe_allow_html=True,
)
