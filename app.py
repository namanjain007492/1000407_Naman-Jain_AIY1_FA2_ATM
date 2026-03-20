"""
╔══════════════════════════════════════════════════════════════════════╗
║     ATM CASH DEMAND FORECASTING & INTELLIGENT DECISION SYSTEM       ║
║                      COMPLETE SINGLE-FILE SOLUTION                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Modules (all-in-one):                                              ║
║  1. Data Preprocessing        5. Anomaly Detection                  ║
║  2. Feature Engineering       6. Prediction Models (RF + GBM)       ║
║  3. Exploratory Data Analysis 7. Decision Alert System              ║
║  4. K-Means Clustering        8. PDF Report + CSV Export            ║
║                                                                      ║
║  Run:  python atm_system.py                                         ║
║  Dashboard: streamlit run atm_system.py                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.cluster          import KMeans
from sklearn.metrics          import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition    import PCA
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection  import train_test_split

from reportlab.lib.pagesizes  import A4
from reportlab.lib            import colors
from reportlab.lib.units      import cm
from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums      import TA_CENTER, TA_LEFT
from reportlab.lib.colors     import HexColor
from reportlab.platypus       import (SimpleDocTemplate, Paragraph, Spacer,
                                       Table, TableStyle, Image, HRFlowable,
                                       PageBreak, KeepTogether)

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════════
DATA_PATH  = "/mnt/user-data/uploads/atm_cash_management_dataset.csv"
OUT_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atm_output")
DPI        = 150
FMT        = FuncFormatter(lambda x, _: f"₹{x/1_000:.0f}K")
DAY_ORDER  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# Brand palette
C_RED   = "#E63946"
C_BLUE  = "#457B9D"
C_GREEN = "#2D6A4F"
C_ORG   = "#F4A261"
C_LIGHT = "#F1FAEE"

CLUSTER_COLORS = {"High Demand": C_RED, "Medium Demand": C_ORG, "Low Demand": C_BLUE}

sns.set_theme(style="whitegrid", font_scale=1.0)

def _mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _savefig(fig, subdir, name):
    path = os.path.join(_mkdir(os.path.join(OUT_ROOT, subdir)), name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def _banner(text):
    w = 62
    print("\n" + "█"*w + f"\n  {text}\n" + "█"*w + "\n")


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 1 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"[LOAD] {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def handle_missing(df):
    """Numeric → median · Categorical → mode"""
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include=["object","string"]).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode()[0], inplace=True)
    print(f"[MISSING] Remaining nulls after fill: {df.isnull().sum().sum()}")
    return df


def parse_dates(df, col="Date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df.sort_values([col, "ATM_ID"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[DATE] Range: {df[col].min().date()} → {df[col].max().date()}")
    return df


def encode_categoricals(df):
    """
    Label-Encode: ATM_ID, Day_of_Week, Time_of_Day, Weather_Condition
    One-Hot-Encode: Location_Type  (nominal, low-cardinality)
    """
    df = df.copy()
    encoders = {}
    for col in ["ATM_ID","Day_of_Week","Time_of_Day","Weather_Condition"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col+"_Enc"] = le.fit_transform(df[col].astype(str))
            encoders[col]  = le
    if "Location_Type" in df.columns:
        dummies = pd.get_dummies(df["Location_Type"], prefix="Location_Type",
                                  drop_first=False, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        encoders["Location_Type"] = list(dummies.columns)
    print(f"[ENCODE] Encoded: {list(encoders.keys())}")
    return df, encoders


def scale_features(df):
    """StandardScaler on 4 core numeric columns (appends *_Scaled)."""
    df   = df.copy()
    cols = [c for c in ["Total_Withdrawals","Total_Deposits",
                          "Previous_Day_Cash_Level","Nearby_Competitor_ATMs"]
            if c in df.columns]
    sc   = StandardScaler()
    df[[c+"_Scaled" for c in cols]] = sc.fit_transform(df[cols])
    print(f"[SCALE] StandardScaler → {cols}")
    return df, sc


def preprocess(filepath):
    print("\n" + "="*60 + "\n  STEP 1 — PREPROCESSING\n" + "="*60)
    df = load_data(filepath)
    df = handle_missing(df)
    df = parse_dates(df)
    df, encoders = encode_categoricals(df)
    df, scaler   = scale_features(df)
    print(f"[DONE] Shape after preprocessing: {df.shape}\n")
    return df, {"encoders": encoders, "scaler": scaler}


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 2 — FEATURE ENGINEERING  (35+ features)
# ═══════════════════════════════════════════════════════════════════════

def _pct_change_safe(s):
    return s.pct_change().replace([np.inf,-np.inf], 0).fillna(0)

def _rolling_slope(series, w=7):
    out = pd.Series(0.0, index=series.index)
    arr = series.values
    for i in range(w-1, len(arr)):
        y = arr[i-w+1:i+1]
        out.iloc[i] = np.polyfit(range(w), y, 1)[0] if np.std(y) > 0 else 0.0
    return out


def add_time_features(df):
    """F1–F9: Calendar signals — weekends, salary dates, seasons."""
    df = df.copy()
    dt = df["Date"]
    df["Day_of_Month"]   = dt.dt.day
    df["Month"]          = dt.dt.month
    df["Quarter"]        = dt.dt.quarter
    df["Day_of_Year"]    = dt.dt.day_of_year
    df["Year"]           = dt.dt.year
    df["Is_Weekend"]     = dt.dt.dayofweek.isin([5,6]).astype(int)
    df["Is_Month_Start"] = dt.dt.is_month_start.astype(int)
    df["Is_Month_End"]   = dt.dt.is_month_end.astype(int)
    sm = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
    df["Season"]      = df["Month"].map(sm)
    df["Season_Name"] = df["Season"].map({1:"Winter",2:"Spring",3:"Summer",4:"Autumn"})
    return df


def add_rolling_features(df):
    """F10–F14: Short & medium rolling mean/std per ATM."""
    df = df.copy().sort_values(["ATM_ID","Date"])
    for col, label in [("Total_Withdrawals","Withdrawals"),
                        ("Total_Deposits","Deposits")]:
        g = df.groupby("ATM_ID")[col]
        df[f"Rolling_3Day_{label}"] = g.transform(lambda x: x.rolling(3,  min_periods=1).mean())
        df[f"Rolling_7Day_{label}"] = g.transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df["Rolling_7Day_Std"] = (
        df.groupby("ATM_ID")["Total_Withdrawals"]
          .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    )
    return df


def add_trend_features(df):
    """F15–F17: Withdrawal/deposit growth rates and 7-day OLS slope."""
    df = df.copy().sort_values(["ATM_ID","Date"])
    df["Withdrawal_Growth_Rate"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(_pct_change_safe)
    df["Deposit_Growth_Rate"]    = df.groupby("ATM_ID")["Total_Deposits"].transform(_pct_change_safe)
    df["Demand_Trend_Slope"]     = (
        df.groupby("ATM_ID")["Total_Withdrawals"]
          .transform(lambda x: _rolling_slope(x, 7).fillna(0))
    )
    return df


def add_ratio_features(df):
    """F18–F21: Business ratios — balance, utilization, risk, excess cash."""
    df  = df.copy()
    eps = 1e-6
    df["Withdrawal_Deposit_Ratio"] = df["Total_Withdrawals"] / (df["Total_Deposits"] + eps)
    df["Cash_Utilization_Rate"]    = (df["Total_Withdrawals"] / (df["Previous_Day_Cash_Level"] + eps)).clip(0,5)
    df["Demand_to_Cash_Ratio"]     = (df["Cash_Demand_Next_Day"] / (df["Previous_Day_Cash_Level"] + eps)).clip(0,5)
    df["Excess_Cash_Level"]        = (df["Previous_Day_Cash_Level"] - df["Total_Withdrawals"]).clip(lower=0)
    return df


def add_event_features(df):
    """F22–F25: Holiday/event impact scores and days-since/until proximity."""
    df = df.copy().sort_values("Date")
    df["Holiday_Impact_Score"] = df["Holiday_Flag"]       * 1.5
    df["Event_Impact_Score"]   = df["Special_Event_Flag"] * 1.25

    hol_dates = df.loc[df["Holiday_Flag"]==1, "Date"].values

    def _since(d):
        past = hol_dates[hol_dates <= d]
        return int((d - past[-1]) / np.timedelta64(1,"D")) if len(past) else -1

    def _until(d):
        fut  = hol_dates[hol_dates >= d]
        return int((fut[0]  - d) / np.timedelta64(1,"D")) if len(fut)  else -1

    darr = df["Date"].values
    df["Days_Since_Last_Holiday"] = [_since(d) for d in darr]
    df["Days_Until_Next_Event"]   = [_until(d) for d in darr]
    return df


def add_location_features(df):
    """F26–F28: Competitor density, urban demand index, location risk score."""
    df = df.copy()
    df["Competitor_Density_Level"] = pd.cut(
        df["Nearby_Competitor_ATMs"], bins=[-1,0,2,100], labels=[0,1,2]
    ).astype(int)
    urban_w = {"Mall":1.0,"Bank Branch":0.9,"Supermarket":0.7,
               "Standalone":0.5,"Gas Station":0.4}
    df["Urban_Demand_Index"] = df["Location_Type"].map(urban_w).fillna(0.5)
    df["Location_Risk_Score"] = (
        df["Competitor_Density_Level"] * 0.4
        + df["Cash_Utilization_Rate"].clip(0,3) * 0.6
    )
    return df


def add_lag_features(df):
    """F29–F31: 1-day, 2-day, 7-day withdrawal lags per ATM."""
    df = df.copy().sort_values(["ATM_ID","Date"])
    for lag in [1,2,7]:
        col = f"Withdrawal_Lag_{lag}"
        df[col] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(lambda x: x.shift(lag))
    lag_cols = ["Withdrawal_Lag_1","Withdrawal_Lag_2","Withdrawal_Lag_7"]
    df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    return df


def add_volatility_flags(df):
    """F32–F34: Demand volatility, peak-hour flag, low-activity flag."""
    df = df.copy()
    df["Demand_Volatility"] = (
        df.groupby("ATM_ID")["Total_Withdrawals"]
          .transform(lambda x: (x.std()/(x.mean()+1e-6)))
          .fillna(0)
    )
    q75 = df["Total_Withdrawals"].quantile(0.75)
    q20 = df["Total_Withdrawals"].quantile(0.20)
    df["Peak_Hour_Flag"]    = ((df["Time_of_Day"].isin(["Evening","Morning"])) &
                               (df["Total_Withdrawals"] > q75)).astype(int)
    df["Low_Activity_Flag"] = (df["Total_Withdrawals"] < q20).astype(int)
    return df


def build_features(df):
    print("\n" + "="*60 + "\n  STEP 2 — FEATURE ENGINEERING\n" + "="*60)
    orig = df.shape[1]
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_ratio_features(df)
    df = add_event_features(df)
    df = add_location_features(df)
    df = add_lag_features(df)
    df = add_volatility_flags(df)
    print(f"[DONE] {orig} → {df.shape[1]} columns (+{df.shape[1]-orig} engineered features)\n")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 3 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def eda_distributions(df, out):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    for ax,(col,c,lbl) in zip(axes,[
        ("Total_Withdrawals",  C_RED,  "Withdrawals"),
        ("Total_Deposits",     C_BLUE, "Deposits"),
    ]):
        d = df[col].dropna()
        ax.hist(d, bins=45, color=c, alpha=0.72, edgecolor="white", density=True)
        d.plot.kde(ax=ax, color="black", lw=2)
        ax.axvline(d.mean(),   color="gold",  lw=2, ls="--", label=f"Mean  ₹{d.mean()/1e3:.1f}K")
        ax.axvline(d.median(), color="lime",  lw=2, ls=":",  label=f"Median ₹{d.median()/1e3:.1f}K")
        ax.set_title(f"{lbl} Distribution", fontweight="bold")
        ax.xaxis.set_major_formatter(FMT); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    fig.suptitle("💰 Cash Flow Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "01_distributions.png")
    skw = df["Total_Withdrawals"].skew()
    print(f"  [EDA-1] Distributions saved.\n"
          f"  INSIGHT: Withdrawal skewness={skw:.2f}. Tail events pull mean above median — "
          "ATM refill plans must account for demand spikes beyond average.\n")
    return p


def eda_timeseries(df, out):
    daily = df.groupby("Date")[["Total_Withdrawals","Total_Deposits",
                                  "Cash_Demand_Next_Day"]].mean().reset_index()
    fig, axes = plt.subplots(3,1, figsize=(16,10), sharex=True)
    cfgs = [("Total_Withdrawals",  C_RED,   "Avg Daily Withdrawals"),
            ("Total_Deposits",     C_BLUE,  "Avg Daily Deposits"),
            ("Cash_Demand_Next_Day",C_GREEN,"Avg Next-Day Cash Demand")]
    for ax,(col,c,title) in zip(axes, cfgs):
        ax.plot(daily["Date"], daily[col], color=c, lw=1.1, alpha=0.75)
        ma = daily[col].rolling(30, min_periods=1).mean()
        ax.plot(daily["Date"], ma, color="black", lw=2, ls="--", label="30-day MA")
        ax.fill_between(daily["Date"], daily[col], alpha=0.1, color=c)
        ax.yaxis.set_major_formatter(FMT)
        ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("📈 ATM Cash Flow — Time Series", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "02_time_series.png")
    print(f"  [EDA-2] Time Series saved.\n"
          "  INSIGHT: 30-day MA reveals gradual demand growth through 2023. "
          "Month-start salary credits and festive-season spikes visible.\n")
    return p


def eda_weekday(df, out):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    day_m = (df[df["Day_of_Week"].isin(DAY_ORDER)]
             .groupby("Day_of_Week")["Total_Withdrawals"].mean()
             .reindex(DAY_ORDER))
    bar_c = [C_RED if d in ["Saturday","Sunday"] else C_BLUE for d in DAY_ORDER]
    axes[0].bar(DAY_ORDER, day_m.values, color=bar_c, edgecolor="white", width=0.6)
    axes[0].yaxis.set_major_formatter(FMT); axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Avg Withdrawal by Day of Week", fontweight="bold")

    df2 = df.copy()
    df2["Day_Type"] = df2["Is_Weekend"].map({1:"Weekend",0:"Weekday"})
    sns.boxplot(data=df2, x="Day_Type", y="Total_Withdrawals",
                palette={"Weekend":C_RED,"Weekday":C_BLUE},
                ax=axes[1], width=0.45)
    axes[1].yaxis.set_major_formatter(FMT)
    axes[1].set_title("Weekday vs Weekend Distribution", fontweight="bold")
    fig.suptitle("📅 Weekday vs Weekend Behavior", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "03_weekday_weekend.png")
    wd = df[df["Is_Weekend"]==0]["Total_Withdrawals"].mean()
    we = df[df["Is_Weekend"]==1]["Total_Withdrawals"].mean()
    print(f"  [EDA-3] Weekday/Weekend saved.\n"
          f"  INSIGHT: Weekend ₹{we/1e3:.1f}K vs Weekday ₹{wd/1e3:.1f}K "
          f"({(we/wd-1)*100:+.1f}%). Stock ATMs Thursday evening.\n")
    return p


def eda_boxplots(df, out):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    sns.boxplot(data=df, x="Location_Type", y="Total_Withdrawals",
                palette="Set2", ax=axes[0], width=0.5,
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
    axes[0].set_title("Withdrawals by Location Type", fontweight="bold")
    axes[0].yaxis.set_major_formatter(FMT); axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(data=df, x="Weather_Condition", y="Total_Withdrawals",
                palette="Set3", ax=axes[1], width=0.5,
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
    axes[1].set_title("Withdrawals by Weather Condition", fontweight="bold")
    axes[1].yaxis.set_major_formatter(FMT)
    fig.suptitle("📦 Outlier Detection via Boxplots", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "04_boxplots.png")
    print("  [EDA-4] Boxplots saved.\n"
          "  INSIGHT: Mall ATMs — widest IQR → highest variability. "
          "Snowy weather suppresses footfall. Outlier dots = anomaly candidates.\n")
    return p


def eda_correlation(df, out):
    key = [c for c in [
        "Total_Withdrawals","Total_Deposits","Previous_Day_Cash_Level",
        "Cash_Demand_Next_Day","Holiday_Flag","Special_Event_Flag",
        "Nearby_Competitor_ATMs","Is_Weekend","Month",
        "Rolling_7Day_Withdrawals","Withdrawal_Lag_1",
        "Urban_Demand_Index","Cash_Utilization_Rate",
        "Demand_Volatility","Withdrawal_Growth_Rate"
    ] if c in df.columns]
    corr = df[key].corr()
    fig, ax = plt.subplots(figsize=(14,11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size":7.5},
                cbar_kws={"shrink":0.75})
    ax.set_title("🔗 Feature Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "05_correlation.png")
    top = corr["Cash_Demand_Next_Day"].abs().sort_values(ascending=False).head(5)
    print(f"  [EDA-5] Correlation saved.\n"
          f"  INSIGHT: Top 5 predictors of demand: {top.index.tolist()}\n"
          "  Rolling averages + lag features → strongest autocorrelation signal.\n")
    return p


def eda_holiday_impact(df, out):
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    for ax,(flag,labels,title,cs) in zip(axes,[
        ("Holiday_Flag",       ["Normal","Holiday"],      "Holiday Impact",       [C_BLUE,C_RED]),
        ("Special_Event_Flag", ["No Event","Special Event"],"Event Impact",        [C_BLUE,C_ORG]),
    ]):
        grp = df.groupby(flag)["Total_Withdrawals"].agg(["mean","std"]).reset_index()
        bars = ax.bar(labels[:len(grp)], grp["mean"], color=cs[:len(grp)],
                      edgecolor="white", width=0.45,
                      yerr=grp["std"], capsize=5,
                      error_kw={"ecolor":"gray","elinewidth":1.5})
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+500,
                    f"₹{b.get_height()/1e3:.1f}K", ha="center", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(FMT)
        ax.set_ylim(0, grp["mean"].max()*1.3)
        ax.set_title(title, fontweight="bold")
    fig.suptitle("🎉 Holiday & Event Impact on Withdrawals", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "06_holiday_event.png")
    h = df[df["Holiday_Flag"]==1]["Total_Withdrawals"].mean()
    n = df[df["Holiday_Flag"]==0]["Total_Withdrawals"].mean()
    print(f"  [EDA-6] Holiday impact saved.\n"
          f"  INSIGHT: Holiday days → {(h/n-1)*100:+.1f}% vs normal. "
          "High error bars = unpredictable spikes → buffer stock essential.\n")
    return p


def eda_external_factors(df, out):
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    wm = df.groupby("Weather_Condition")["Total_Withdrawals"].mean().sort_values(ascending=False)
    axes[0].barh(wm.index, wm.values,
                 color=[C_RED,C_ORG,C_BLUE,C_GREEN][:len(wm)], edgecolor="white")
    axes[0].xaxis.set_major_formatter(FMT)
    axes[0].set_title("Avg Withdrawals by Weather", fontweight="bold")
    for i,(idx,val) in enumerate(wm.items()):
        axes[0].text(val+200, i, f"₹{val/1e3:.1f}K", va="center", fontsize=8)

    cm_ = df.groupby("Nearby_Competitor_ATMs")["Total_Withdrawals"].mean()
    axes[1].plot(cm_.index, cm_.values, "o-", color=C_RED, lw=2, ms=7)
    axes[1].fill_between(cm_.index, cm_.values, alpha=0.15, color=C_RED)
    axes[1].yaxis.set_major_formatter(FMT)
    axes[1].set_title("Withdrawals vs Competitor Count", fontweight="bold")
    axes[1].set_xlabel("Nearby Competitor ATMs")
    fig.suptitle("🌦️ Weather & Competition Effects", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "07_external_factors.png")
    print("  [EDA-7] External factors saved.\n"
          "  INSIGHT: Clear weather → max footfall. "
          "More competitors → demand split across machines.\n")
    return p


def eda_demand_heatmap(df, out):
    pivot = df.pivot_table(values="Cash_Demand_Next_Day",
                            index="Location_Type", columns="Day_of_Week",
                            aggfunc="mean")
    pivot = pivot.reindex(columns=[d for d in DAY_ORDER if d in pivot.columns])
    fig, ax = plt.subplots(figsize=(13,5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, annot_kws={"size":9},
                cbar_kws={"label":"Avg Demand (₹)"})
    ax.set_title("🗺️ Cash Demand Heatmap — Location × Day", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "eda", "08_demand_heatmap.png")
    print("  [EDA-8] Demand heatmap saved.\n"
          "  INSIGHT: Mall/Friday & Mall/Saturday = highest demand. "
          "Gas Station/Sunday = lowest. Tailor stock per cell.\n")
    return p


def run_eda(df):
    print("\n" + "="*60 + "\n  STEP 3 — EXPLORATORY DATA ANALYSIS\n" + "="*60)
    out = _mkdir(os.path.join(OUT_ROOT,"eda"))
    paths = {
        "distributions":   eda_distributions(df, out),
        "time_series":     eda_timeseries(df, out),
        "weekday_weekend": eda_weekday(df, out),
        "boxplots":        eda_boxplots(df, out),
        "correlation":     eda_correlation(df, out),
        "holiday_impact":  eda_holiday_impact(df, out),
        "external":        eda_external_factors(df, out),
        "heatmap":         eda_demand_heatmap(df, out),
    }
    print(f"[DONE] EDA: {len(paths)} charts → {out}\n")
    return paths


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 4 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════

CLUST_FEATS = [
    "Total_Withdrawals","Total_Deposits","Previous_Day_Cash_Level",
    "Rolling_7Day_Withdrawals","Cash_Utilization_Rate","Urban_Demand_Index",
    "Withdrawal_Deposit_Ratio","Withdrawal_Lag_1","Holiday_Impact_Score",
    "Demand_Volatility","Withdrawal_Growth_Rate"
]

def _prepare_cluster_data(df):
    feats = [f for f in CLUST_FEATS if f in df.columns]
    X     = df[feats].fillna(df[feats].median()).values
    Xsc   = StandardScaler().fit_transform(X)
    return Xsc, feats


def _elbow_optimal_k(Xsc, k_range=range(2,10)):
    inertias, silhs = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        lbl = km.fit_predict(Xsc)
        inertias.append(km.inertia_)
        silhs.append(silhouette_score(Xsc, lbl,
                                       sample_size=min(1000, Xsc.shape[0]),
                                       random_state=42))
    d2    = np.diff(np.diff(inertias))
    opt_k = max(int(list(k_range)[np.argmax(d2)+1]), 3)

    fig, ax1 = plt.subplots(figsize=(10,5))
    ks = list(k_range)
    ax1.plot(ks, inertias, "bo-", lw=2, ms=6, label="Inertia (WCSS)")
    ax1.axvline(opt_k, color="red", ls="--", lw=1.8, label=f"Optimal k={opt_k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(ks, silhs, "rs--", lw=1.5, ms=5, alpha=0.8, label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)
    plt.title(f"Elbow + Silhouette  (optimal k={opt_k})", fontweight="bold")
    fig.tight_layout()
    ep = _savefig(fig, "clustering", "09_elbow_silhouette.png")
    print(f"  [CLUST] Elbow → optimal k={opt_k} | Silhouette={silhs[ks.index(opt_k)]:.4f}")
    return opt_k, ep, silhs


def _assign_cluster_labels(df, raw_labels):
    df = df.copy()
    df["_raw"] = raw_labels
    order = (df.groupby("_raw")["Total_Withdrawals"]
               .mean().sort_values(ascending=False).index.tolist())
    names = ["High Demand","Medium Demand","Low Demand"]
    names += [f"Tier {i+4}" for i in range(max(0, len(order)-3))]
    label_map = {cid: names[i] for i, cid in enumerate(order)}
    df["Cluster_ID"]    = df["_raw"].map({v:i for i,v in enumerate(order)})
    df["Cluster_Label"] = df["_raw"].map(label_map)
    df.drop(columns="_raw", inplace=True)
    return df


def _plot_clusters(df, Xsc):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xsc)
    var    = pca.explained_variance_ratio_ * 100
    dfp    = df.copy(); dfp["PC1"] = coords[:,0]; dfp["PC2"] = coords[:,1]

    fig, axes = plt.subplots(1,2, figsize=(15,6))
    for lbl, grp in dfp.groupby("Cluster_Label"):
        axes[0].scatter(grp["PC1"], grp["PC2"],
                        c=CLUSTER_COLORS.get(lbl, "#888"),
                        label=lbl, alpha=0.5, s=14, edgecolors="none")
    axes[0].set_xlabel(f"PC1 ({var[0]:.1f}% var)"); axes[0].set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    axes[0].set_title("K-Means Clusters (PCA 2-D)", fontweight="bold"); axes[0].legend(fontsize=9)

    cm_w = dfp.groupby("Cluster_Label")["Total_Withdrawals"].mean()
    axes[1].bar(cm_w.index, cm_w.values,
                color=[CLUSTER_COLORS.get(l,"#888") for l in cm_w.index],
                edgecolor="white", width=0.5)
    axes[1].yaxis.set_major_formatter(FMT)
    axes[1].set_title("Mean Withdrawals per Cluster", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    for i,(idx,v) in enumerate(cm_w.items()):
        axes[1].text(i, v+300, f"₹{v/1e3:.1f}K", ha="center", fontweight="bold")

    fig.suptitle("🔵 ATM Demand Clusters", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "clustering", "10_clusters_pca.png")


def _plot_cluster_comparison(df):
    metrics = [c for c in ["Total_Withdrawals","Cash_Utilization_Rate",
                            "Urban_Demand_Index","Demand_Volatility",
                            "Rolling_7Day_Withdrawals","Withdrawal_Deposit_Ratio"]
               if c in df.columns][:6]
    fig, axes = plt.subplots(2,3, figsize=(16,9))
    axes = axes.flatten()
    for ax, m in zip(axes, metrics):
        vals = df.groupby("Cluster_Label")[m].mean()
        ax.bar(vals.index, vals.values,
               color=[CLUSTER_COLORS.get(l,"#888") for l in vals.index],
               edgecolor="white", width=0.5)
        ax.set_title(m, fontweight="bold", fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        if any(k in m for k in ["Withdrawal","Cash_Level","Rolling"]):
            ax.yaxis.set_major_formatter(FMT)
    for ax in axes[len(metrics):]: ax.set_visible(False)
    fig.suptitle("📊 Cluster Comparison Dashboard", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "clustering", "11_cluster_comparison.png")


def run_clustering(df):
    print("\n" + "="*60 + "\n  STEP 4 — CLUSTERING\n" + "="*60)
    Xsc, feats = _prepare_cluster_data(df)
    opt_k, elbow_p, silhs = _elbow_optimal_k(Xsc)

    km     = KMeans(n_clusters=opt_k, init="k-means++", n_init=20, random_state=42)
    labels = km.fit_predict(Xsc)
    df     = _assign_cluster_labels(df, labels)

    scatter_p    = _plot_clusters(df, Xsc)
    comparison_p = _plot_cluster_comparison(df)

    print("\n  Cluster Statistics:")
    grp = df.groupby("Cluster_Label")[["Total_Withdrawals","Cash_Utilization_Rate",
                                        "Urban_Demand_Index"]].mean()
    for lbl, row in grp.iterrows():
        print(f"  🔹 {lbl:15s} | Avg Withdrawal: ₹{row['Total_Withdrawals']/1e3:.1f}K "
              f"| Utilization: {row['Cash_Utilization_Rate']:.2f}x "
              f"| Urban Index: {row['Urban_Demand_Index']:.2f}")
    print(f"\n[DONE] Clustering complete. Labels: {df['Cluster_Label'].unique().tolist()}\n")

    return df, {
        "model": km, "features": feats,
        "paths": {"elbow": elbow_p, "scatter": scatter_p, "comparison": comparison_p}
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 5 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_zscore(df, threshold=3.0):
    """Per-ATM z-score. |z| > threshold → anomaly."""
    df = df.copy()
    df["Zscore_Withdrawal"] = (
        df.groupby("ATM_ID")["Total_Withdrawals"]
          .transform(lambda x: scipy_stats.zscore(x, nan_policy="omit"))
          .fillna(0)
    )
    df["Is_Anomaly_ZScore"] = (df["Zscore_Withdrawal"].abs() > threshold).astype(int)
    n = df["Is_Anomaly_ZScore"].sum()
    print(f"  [ANOM-Z] Z-score (>{threshold}σ): {n} anomalies ({n/len(df)*100:.1f}%)")
    return df


def detect_iqr(df, mult=1.5):
    """Per-ATM IQR fence. Below Q1-k*IQR or above Q3+k*IQR → anomaly."""
    df = df.copy()
    def _flag(s):
        q1,q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3-q1
        return ((s < q1-mult*iqr) | (s > q3+mult*iqr)).astype(int)
    df["Is_Anomaly_IQR"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(_flag)
    n = df["Is_Anomaly_IQR"].sum()
    print(f"  [ANOM-IQR] IQR (×{mult}): {n} anomalies ({n/len(df)*100:.1f}%)")
    return df


def detect_isoforest(df, contamination=0.05):
    """Multi-feature Isolation Forest for contextual anomalies."""
    df = df.copy()
    fcols = [c for c in ["Total_Withdrawals","Total_Deposits","Previous_Day_Cash_Level",
                           "Zscore_Withdrawal","Cash_Utilization_Rate",
                           "Rolling_7Day_Withdrawals","Demand_Volatility"] if c in df.columns]
    X   = df[fcols].fillna(df[fcols].median()).values
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                           random_state=42, n_jobs=-1)
    df["Is_Anomaly_IsoForest"] = (iso.fit_predict(X) == -1).astype(int)
    n = df["Is_Anomaly_IsoForest"].sum()
    print(f"  [ANOM-ISO] Isolation Forest: {n} anomalies ({n/len(df)*100:.1f}%)")
    return df, iso


def build_consensus(df, min_votes=2):
    """Flag rows agreed upon by ≥ min_votes methods."""
    df   = df.copy()
    vcols = [c for c in ["Is_Anomaly_ZScore","Is_Anomaly_IQR","Is_Anomaly_IsoForest"]
             if c in df.columns]
    df["Anomaly_Votes"] = df[vcols].sum(axis=1)
    df["Is_Anomaly"]    = (df["Anomaly_Votes"] >= min_votes).astype(int)
    n = df["Is_Anomaly"].sum()
    print(f"  [ANOM] Consensus (≥{min_votes} votes): {n} anomalies ({n/len(df)*100:.1f}%)")
    return df


def _plot_anomaly_timeseries(df):
    daily = df.groupby("Date").agg(
        W=("Total_Withdrawals","mean"), A=("Is_Anomaly","max")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(16,5))
    ax.plot(daily["Date"], daily["W"], color=C_BLUE, lw=1.1, alpha=0.8, label="Daily Avg")
    ma = daily["W"].rolling(14, min_periods=1).mean()
    ax.plot(daily["Date"], ma, color="black", lw=2, ls="--", alpha=0.7, label="14-day MA")
    an = daily[daily["A"]==1]
    ax.scatter(an["Date"], an["W"], color=C_RED, s=55, zorder=5, label="Anomaly Day")
    for _, row in an.iterrows():
        ax.axvspan(row["Date"]-pd.Timedelta(hours=12),
                   row["Date"]+pd.Timedelta(hours=12), alpha=0.18, color=C_RED)
    ax.yaxis.set_major_formatter(FMT); ax.legend(fontsize=9)
    ax.set_title("🚨 Anomaly Detection — Time Series", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "anomaly", "12_anomaly_timeseries.png")


def _plot_anomaly_overview(df):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    mc = {
        "Z-Score":          df.get("Is_Anomaly_ZScore",   pd.Series(dtype=int)).sum(),
        "IQR":              df.get("Is_Anomaly_IQR",      pd.Series(dtype=int)).sum(),
        "Isolation Forest": df.get("Is_Anomaly_IsoForest",pd.Series(dtype=int)).sum(),
        "Consensus":        df.get("Is_Anomaly",          pd.Series(dtype=int)).sum(),
    }
    bars = axes[0].bar(mc.keys(), mc.values(),
                       color=[C_ORG,C_GREEN,C_BLUE,C_RED], edgecolor="white", width=0.55)
    for b in bars:
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     str(int(b.get_height())), ha="center", fontweight="bold")
    axes[0].set_title("Anomaly Count by Method", fontweight="bold")
    axes[0].set_ylabel("Count")

    norm = df[df["Is_Anomaly"]==0]["Total_Withdrawals"]
    anom = df[df["Is_Anomaly"]==1]["Total_Withdrawals"]
    axes[1].hist(norm, bins=40, color=C_BLUE, alpha=0.65, density=True, label="Normal")
    axes[1].hist(anom, bins=20, color=C_RED,  alpha=0.75, density=True, label="Anomaly")
    axes[1].xaxis.set_major_formatter(FMT)
    axes[1].set_title("Normal vs Anomaly Distribution", fontweight="bold")
    axes[1].legend()
    fig.suptitle("🔍 Anomaly Detection Overview", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "anomaly", "13_anomaly_overview.png")


def _plot_anomaly_context(df):
    an = df[df["Is_Anomaly"]==1].copy()
    if len(an) == 0: return None
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    lc = an["Location_Type"].value_counts()
    axes[0].barh(lc.index, lc.values, color=C_RED, edgecolor="white")
    axes[0].set_title("Anomalies by Location", fontweight="bold"); axes[0].set_xlabel("Count")

    nr = df[df["Is_Anomaly"]==0]
    fd = pd.DataFrame({
        "Holiday":       [an["Holiday_Flag"].sum(),       nr["Holiday_Flag"].sum()],
        "Special Event": [an["Special_Event_Flag"].sum(), nr["Special_Event_Flag"].sum()],
    }, index=["Anomaly","Normal"]).T
    fd.plot(kind="bar", ax=axes[1], color=[C_RED,C_BLUE], edgecolor="white", width=0.5)
    axes[1].set_title("Holiday/Event in Anomaly vs Normal", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=0); axes[1].legend(fontsize=8)

    dc = an["Day_of_Week"].value_counts().reindex(DAY_ORDER).dropna()
    axes[2].bar(dc.index, dc.values, color=C_ORG, edgecolor="white")
    axes[2].set_title("Anomalies by Day of Week", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle("📍 Anomaly Context — Why Did It Happen?", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "anomaly", "14_anomaly_context.png")
    hpct = an["Holiday_Flag"].mean()*100
    epct = an["Special_Event_Flag"].mean()*100
    print(f"\n  INSIGHT: Of {len(an)} anomalies — {hpct:.0f}% on holidays, "
          f"{epct:.0f}% on event days. Remaining may be hardware/data issues.")
    return p


def run_anomaly_detection(df):
    print("\n" + "="*60 + "\n  STEP 5 — ANOMALY DETECTION\n" + "="*60)
    df = detect_zscore(df)
    df = detect_iqr(df)
    df, iso = detect_isoforest(df)
    df = build_consensus(df)
    paths = {
        "timeseries": _plot_anomaly_timeseries(df),
        "overview":   _plot_anomaly_overview(df),
        "context":    _plot_anomaly_context(df),
    }
    print(f"\n[DONE] Anomaly detection complete. Total flagged: {df['Is_Anomaly'].sum()}\n")
    return df, {"model": iso, "paths": paths}


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 6 — PREDICTION MODELS
# ═══════════════════════════════════════════════════════════════════════

MODEL_FEATURES = [
    "Total_Withdrawals","Total_Deposits","Previous_Day_Cash_Level",
    "Is_Weekend","Month","Quarter","Day_of_Year","Season",
    "Is_Month_Start","Is_Month_End","Day_of_Month",
    "Rolling_3Day_Withdrawals","Rolling_7Day_Withdrawals",
    "Rolling_3Day_Deposits","Rolling_7Day_Deposits","Rolling_7Day_Std",
    "Withdrawal_Growth_Rate","Deposit_Growth_Rate","Demand_Trend_Slope",
    "Withdrawal_Deposit_Ratio","Cash_Utilization_Rate","Demand_to_Cash_Ratio",
    "Holiday_Flag","Special_Event_Flag","Holiday_Impact_Score",
    "Event_Impact_Score","Days_Since_Last_Holiday",
    "Nearby_Competitor_ATMs","Competitor_Density_Level",
    "Urban_Demand_Index","Location_Risk_Score",
    "Withdrawal_Lag_1","Withdrawal_Lag_2","Withdrawal_Lag_7",
    "Demand_Volatility","Peak_Hour_Flag","Low_Activity_Flag",
    "Is_Anomaly","Anomaly_Votes","Zscore_Withdrawal",
    "ATM_ID_Enc","Day_of_Week_Enc","Time_of_Day_Enc","Weather_Condition_Enc",
    "Location_Type_Bank Branch","Location_Type_Gas Station",
    "Location_Type_Mall","Location_Type_Standalone","Location_Type_Supermarket",
]
TARGET = "Cash_Demand_Next_Day"


def _get_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE":  mean_absolute_error(y_test, preds),
        "R2":   r2_score(y_test, preds),
        "predictions": preds,
    }


def train_models(df):
    avail = [f for f in MODEL_FEATURES if f in df.columns]
    sub   = df[avail + [TARGET]].dropna(subset=[TARGET])
    X     = sub[avail].fillna(sub[avail].median())
    y     = sub[TARGET].values

    split   = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split],       y[split:]
    print(f"  [MODEL] Features used: {len(avail)} | Train: {len(X_tr):,} | Test: {len(X_te):,}")
    print(f"  [MODEL] Target range: ₹{y.min():,.0f} – ₹{y.max():,.0f}")

    rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_m = _get_metrics(rf, X_te, y_te)
    print(f"  [RF]  RMSE=₹{rf_m['RMSE']:,.0f}  MAE=₹{rf_m['MAE']:,.0f}  R²={rf_m['R2']:.4f}")

    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     min_samples_leaf=5, subsample=0.8, random_state=42)
    gbm.fit(X_tr, y_tr)
    gbm_m = _get_metrics(gbm, X_te, y_te)
    print(f"  [GBM] RMSE=₹{gbm_m['RMSE']:,.0f}  MAE=₹{gbm_m['MAE']:,.0f}  R²={gbm_m['R2']:.4f}")

    best_model   = rf  if rf_m["RMSE"] <= gbm_m["RMSE"] else gbm
    best_name    = "RandomForest" if best_model is rf else "GradientBoosting"
    best_metrics = rf_m if best_model is rf else gbm_m
    print(f"\n  ★ Best: {best_name}  R²={best_metrics['R2']:.4f}  RMSE=₹{best_metrics['RMSE']:,.0f}")

    return {
        "models":  {"RandomForest": rf, "GradientBoosting": gbm},
        "metrics": {"RandomForest": rf_m, "GradientBoosting": gbm_m},
        "best":    {"model": best_model, "name": best_name, "metrics": best_metrics},
        "data":    {"X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te},
        "features": avail,
    }


def _plot_feature_importance(art):
    model = art["best"]["model"]; feats = art["features"]; name = art["best"]["name"]
    imp   = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    top20 = imp.head(20)
    fig, axes = plt.subplots(1,2, figsize=(16,7))
    clrs = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top20)))[::-1]
    axes[0].barh(top20.index[::-1], top20.values[::-1], color=clrs, edgecolor="white")
    axes[0].set_xlabel("Importance"); axes[0].set_title(f"Top 20 Features ({name})", fontweight="bold")
    for i,(idx,v) in enumerate(zip(top20.index[::-1], top20.values[::-1])):
        axes[0].text(v+0.0003, i, f"{v:.3f}", va="center", fontsize=7.5)

    cum = imp.cumsum() / imp.sum() * 100
    axes[1].plot(range(1,len(cum)+1), cum.values, "o-", color=C_RED, lw=2, ms=4, alpha=0.8)
    axes[1].axhline(80, color="gray", ls="--", lw=1.5, label="80% threshold")
    axes[1].axhline(95, color="black", ls=":", lw=1.5, label="95% threshold")
    n80 = int((cum < 80).sum())+1
    axes[1].axvline(n80, color=C_ORG, ls="--", lw=1.5, label=f"{n80} features → 80%")
    axes[1].set_xlabel("# Features"); axes[1].set_ylabel("Cumulative Importance (%)")
    axes[1].set_title("Cumulative Feature Importance", fontweight="bold"); axes[1].legend(fontsize=8)
    fig.suptitle("🔬 Feature Importance Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = _savefig(fig, "models", "15_feature_importance.png")
    print(f"  [MODEL] Top 3 features: {top20.index[:3].tolist()}")
    print(f"  [MODEL] {n80} features explain 80% of predictive power.")
    return p


def _plot_actual_vs_predicted(art):
    y_te = art["data"]["y_te"]; preds = art["best"]["metrics"]["predictions"]
    r2   = art["best"]["metrics"]["R2"];   name = art["best"]["name"]
    fig, axes = plt.subplots(1,2, figsize=(15,6))
    axes[0].scatter(y_te, preds, alpha=0.35, s=14, color=C_BLUE, edgecolors="none")
    lo = min(y_te.min(), preds.min()); hi = max(y_te.max(), preds.max())
    axes[0].plot([lo,hi],[lo,hi], "r--", lw=2, label="Perfect")
    axes[0].xaxis.set_major_formatter(FMT); axes[0].yaxis.set_major_formatter(FMT)
    axes[0].set_title(f"Actual vs Predicted  (R²={r2:.4f})", fontweight="bold")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted"); axes[0].legend()

    res = y_te - preds
    axes[1].hist(res, bins=50, color=C_RED, alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="black", lw=2, ls="--", label="Zero error")
    axes[1].axvline(res.mean(), color="gold", lw=2, ls="-",
                    label=f"Mean ₹{res.mean():,.0f}")
    axes[1].xaxis.set_major_formatter(FMT)
    axes[1].set_title("Residual Distribution", fontweight="bold"); axes[1].legend(fontsize=8)
    fig.suptitle(f"📉 {name} — Prediction Evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "models", "16_actual_vs_predicted.png")


def _plot_model_comparison(art):
    rf_m  = art["metrics"]["RandomForest"]
    gbm_m = art["metrics"]["GradientBoosting"]
    fig, axes = plt.subplots(1,3, figsize=(14,5))
    for ax, metric, rv, gv in zip(axes,
        ["RMSE","MAE","R2"],
        [rf_m["RMSE"],rf_m["MAE"],rf_m["R2"]],
        [gbm_m["RMSE"],gbm_m["MAE"],gbm_m["R2"]]):
        bars = ax.bar(["Random Forest","Gradient Boosting"], [rv,gv],
                      color=[C_BLUE,C_RED], edgecolor="white", width=0.45)
        for b in bars:
            fmt = f"₹{b.get_height():,.0f}" if metric in ["RMSE","MAE"] else f"{b.get_height():.4f}"
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01,
                    fmt, ha="center", fontsize=9, fontweight="bold")
        ax.set_title(metric, fontweight="bold"); ax.tick_params(axis="x", rotation=15)
        if metric in ["RMSE","MAE"]: ax.yaxis.set_major_formatter(FMT)
    fig.suptitle("🏆 Model Comparison: RF vs GBM", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "models", "17_model_comparison.png")


def annotate_predictions(df, art):
    model = art["best"]["model"]; feats = art["features"]
    avail = [f for f in feats if f in df.columns]
    X_all = df[avail].fillna(df[avail].median())
    preds = model.predict(X_all)
    df = df.copy()
    df["Predicted_Demand"]  = preds.astype(int)
    df["Prediction_Error"]  = (df[TARGET] - df["Predicted_Demand"]).abs()
    df["Confidence_Score"]  = (1 - df["Prediction_Error"] / (df[TARGET]+1)).clip(0,1).round(4)
    print(f"  [MODEL] Avg confidence: {df['Confidence_Score'].mean():.2%}")
    return df


def run_modeling(df):
    print("\n" + "="*60 + "\n  STEP 6 — PREDICTION MODELS\n" + "="*60)
    art   = train_models(df)
    paths = {
        "importance":          _plot_feature_importance(art),
        "actual_vs_predicted": _plot_actual_vs_predicted(art),
        "comparison":          _plot_model_comparison(art),
    }
    df = annotate_predictions(df, art)
    print(f"\n[DONE] Modeling: best={art['best']['name']}  R²={art['best']['metrics']['R2']:.4f}\n")
    return df, art, paths


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 7 — ALERT SYSTEM & DECISION INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════

BUFFER = 0.20   # 20% safety buffer above predicted demand

def generate_alerts(df):
    """
    Rule-based decision engine:
      CRITICAL  — cash < 50% of predicted demand
      HIGH      — cash < 80% of predicted demand
      MEDIUM    — holiday/event spike OR anomaly detected
      LOW       — excess cash (idle capital)
      NORMAL    — all good
    """
    df   = df.copy()
    pred = "Predicted_Demand" if "Predicted_Demand" in df.columns else TARGET
    p75  = df[pred].quantile(0.75)

    types, sevs, refills, reasons = [], [], [], []
    for _, row in df.iterrows():
        p    = row[pred]; c = row["Previous_Day_Cash_Level"]
        hol  = row.get("Holiday_Flag",0); evt = row.get("Special_Event_Flag",0)
        anom = row.get("Is_Anomaly",0)
        need = max(0, p*(1+BUFFER) - c)

        if c < p * 0.50:
            t,s,r = "🔴 CRITICAL: Cash Shortage","CRITICAL", f"Cash ₹{c:,.0f} < 50% of demand ₹{p:,.0f}"
        elif c < p * 0.80:
            t,s,r = "🟠 HIGH: Low Cash Warning","HIGH",     f"Cash ₹{c:,.0f} < 80% of demand ₹{p:,.0f}"
        elif p > p75 and (hol or evt):
            t,s,r = "🟡 MEDIUM: Holiday Spike","MEDIUM",   f"Surge ₹{p:,.0f} on {'Holiday' if hol else 'Event'} day"
        elif anom:
            t,s,r = "🟡 MEDIUM: Unusual Activity","MEDIUM","Anomaly detected — pattern outside normal range"
        elif c > p * 2.5:
            t,s,r = "🟢 LOW: Excess Cash","LOW",           f"Cash ₹{c:,.0f} = {c/p:.1f}x demand — reduce refill"
            need  = 0
        else:
            t,s,r = "✅ NORMAL","NORMAL","Cash level adequate"

        types.append(t); sevs.append(s); refills.append(round(need)); reasons.append(r)

    df["Alert_Type"]         = types
    df["Alert_Severity"]     = sevs
    df["Recommended_Refill"] = refills
    df["Alert_Reason"]       = reasons

    sev_ct = df["Alert_Severity"].value_counts()
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║         ALERT SYSTEM SUMMARY             ║")
    print("  ╠══════════════════════════════════════════╣")
    for sev, em in [("CRITICAL","🔴"),("HIGH","🟠"),("MEDIUM","🟡"),("LOW","🟢"),("NORMAL","✅")]:
        n = sev_ct.get(sev,0)
        print(f"  ║  {em} {sev:10s}: {n:>5,} records")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  Total Recommended Refill: ₹{df['Recommended_Refill'].sum()/1e6:.2f}M")
    print("  ╚══════════════════════════════════════════╝\n")
    return df


def _plot_alert_dashboard(df):
    sev_colors = {"CRITICAL":C_RED,"HIGH":C_ORG,"MEDIUM":"#FFBA08",
                  "LOW":C_GREEN,"NORMAL":"#4CAF50"}
    fig = plt.figure(figsize=(16,11))
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)

    # Panel 1 — Severity pie
    ax1 = fig.add_subplot(gs[0,0])
    sc  = df["Alert_Severity"].value_counts().reindex(
        ["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"]).dropna()
    ax1.pie(sc.values, labels=sc.index,
            colors=[sev_colors.get(s,"grey") for s in sc.index],
            autopct="%1.1f%%", startangle=140, textprops={"fontsize":9})
    ax1.set_title("🚦 Alert Severity Mix", fontweight="bold")

    # Panel 2 — Avg refill by location
    ax2 = fig.add_subplot(gs[0,1])
    lr  = df.groupby("Location_Type")["Recommended_Refill"].mean().sort_values(ascending=True)
    ax2.barh(lr.index, lr.values, color=C_RED, edgecolor="white")
    ax2.xaxis.set_major_formatter(FMT)
    for i,(idx,v) in enumerate(lr.items()):
        ax2.text(v+100, i, f"₹{v/1e3:.1f}K", va="center", fontsize=8)
    ax2.set_title("💰 Avg Recommended Refill\nby Location", fontweight="bold")

    # Panel 3 — Alert timeline (full width)
    ax3 = fig.add_subplot(gs[1,:])
    daily_a = df.groupby("Date")["Alert_Severity"].apply(
        lambda x: pd.Series({
            "CRITICAL": (x=="CRITICAL").sum(),
            "HIGH":     (x=="HIGH").sum(),
            "MEDIUM":   (x=="MEDIUM").sum(),
        })
    ).reset_index()
    if "level_1" in daily_a.columns:
        daily_a = daily_a.pivot(index="Date", columns="level_1", values="Alert_Severity").reset_index()
    elif isinstance(daily_a.columns, pd.MultiIndex):
        daily_a.columns = ["Date"] + list(daily_a.columns[1:])

    # Safer approach: recompute daily counts directly
    dal = df.groupby("Date").agg(
        CRITICAL=("Alert_Severity", lambda x: (x=="CRITICAL").sum()),
        HIGH    =("Alert_Severity", lambda x: (x=="HIGH").sum()),
        MEDIUM  =("Alert_Severity", lambda x: (x=="MEDIUM").sum()),
    ).reset_index()
    ax3.stackplot(dal["Date"], dal["CRITICAL"], dal["HIGH"], dal["MEDIUM"],
                  labels=["Critical","High","Medium"],
                  colors=[C_RED, C_ORG, "#FFBA08"], alpha=0.85)
    ax3.set_title("📅 Alert Timeline — Daily Count", fontweight="bold")
    ax3.set_xlabel("Date"); ax3.set_ylabel("# Alerts"); ax3.legend(loc="upper left", fontsize=9)

    fig.suptitle("🔔 ATM Cash Management — Alert Dashboard", fontsize=14, fontweight="bold")
    p = _savefig(fig, ".", "18_alert_dashboard.png")
    print(f"  [ALERT] Dashboard → {p}")
    return p


def priority_atm_table(df, n=20):
    order = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"NORMAL":4}
    df2   = df.copy()
    df2["_o"] = df2["Alert_Severity"].map(order)
    cols  = [c for c in ["ATM_ID","Date","Location_Type","Predicted_Demand",
                          "Recommended_Refill","Alert_Type","Alert_Reason"] if c in df2.columns]
    return (df2.sort_values("_o")
               .drop_duplicates("ATM_ID", keep="first")
               .head(n)[cols])


def run_alerts(df):
    print("\n" + "="*60 + "\n  STEP 7 — ALERT SYSTEM\n" + "="*60)
    df    = generate_alerts(df)
    ap    = _plot_alert_dashboard(df)
    tbl   = priority_atm_table(df)
    print(f"[DONE] Alerts complete.\n")
    return df, {"dashboard_path": ap, "report_table": tbl}


# ═══════════════════════════════════════════════════════════════════════
#  MODULE 8 — PDF REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_pdf_report(df, art, eda_paths, cluster_paths, anomaly_paths, alert_art):
    from datetime import datetime
    _mkdir(os.path.join(OUT_ROOT,"reports"))
    path = os.path.join(OUT_ROOT,"reports",
                        f"ATM_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    P  = HexColor("#1A1A2E"); A = HexColor("#E63946")
    MI = HexColor("#A8DADC"); L = HexColor("#F1FAEE")
    SS = getSampleStyleSheet()
    def sty(name, **kw):
        return ParagraphStyle(name, parent=SS["Normal"], **kw)

    H1  = sty("H1", fontName="Helvetica-Bold", fontSize=15, textColor=P, spaceBefore=14, spaceAfter=6)
    H2  = sty("H2", fontName="Helvetica-Bold", fontSize=11, textColor=HexColor("#457B9D"), spaceBefore=8, spaceAfter=4)
    BD  = sty("BD", fontName="Helvetica", fontSize=9, leading=14)
    INS = sty("INS", fontName="Helvetica-Oblique", fontSize=8.5, leading=13,
               textColor=HexColor("#1A6B4A"), leftIndent=10)

    def img(p, w=16*cm, h=8.5*cm):
        return Image(p, width=w, height=h) if p and os.path.exists(p) else Spacer(1,1*cm)

    def tbl_style(data, col_w=None):
        cw = col_w or [19*cm/max(len(data[0]),1)]*len(data[0])
        t  = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),P), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,L]),
            ("GRID",(0,0),(-1,-1),0.3,MI), ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("ROWHEIGHT",(0,0),(-1,-1),0.5*cm),
        ]))
        return t

    story = []
    # Cover
    story += [Spacer(1,2.5*cm),
              Paragraph("ATM CASH DEMAND FORECASTING", sty("cov",fontName="Helvetica-Bold",
                         fontSize=24,textColor=colors.white,alignment=TA_CENTER)),
              Paragraph("Intelligent Decision System — Full Report",
                        sty("sub",fontName="Helvetica",fontSize=13,
                             textColor=HexColor("#A8DADC"),alignment=TA_CENTER)),
              Spacer(1,0.5*cm),
              HRFlowable(width="100%",thickness=2,color=A), Spacer(1,0.4*cm),
              Paragraph(f"Generated: {__import__('datetime').datetime.now().strftime('%d %B %Y %H:%M')}",
                        sty("dt",fontSize=9,textColor=colors.grey,alignment=TA_CENTER)),
              Paragraph(f"Dataset: {len(df):,} records · {df['ATM_ID'].nunique()} ATMs · "
                        f"{df['Date'].min().date()} → {df['Date'].max().date()}",
                        sty("meta",fontSize=10,textColor=HexColor("#457B9D"),alignment=TA_CENTER)),
              PageBreak()]

    # KPIs
    m  = art["best"]["metrics"]
    r2 = m["R2"]; rmse = m["RMSE"]; best = art["best"]["name"][:10]
    na = int(df.get("Is_Anomaly",pd.Series(dtype=int)).sum())
    aw = df["Total_Withdrawals"].mean()
    hol_n = df[df["Holiday_Flag"]==1]["Total_Withdrawals"].mean()
    nrm_n = df[df["Holiday_Flag"]==0]["Total_Withdrawals"].mean()
    crit  = (df.get("Alert_Severity","NORMAL")=="CRITICAL").sum()

    kpi_d = [
        ["₹"+f"{aw/1e3:.1f}K", f"{r2:.4f}", f"₹{rmse/1e3:.1f}K",
         str(na), str(int(crit)), f"{(hol_n/nrm_n-1)*100:+.1f}%"],
        [f"Avg Withdrawal", f"R² ({best})", "RMSE",
         "Anomalies", "Critical Alerts", "Holiday Lift"],
    ]
    kpi_t = Table([[Paragraph(v, sty("k",fontName="Helvetica-Bold",fontSize=16,
                                      textColor=A,alignment=TA_CENTER)) for v in kpi_d[0]],
                   [Paragraph(v, sty("kl",fontSize=8,textColor=colors.grey,
                                      alignment=TA_CENTER)) for v in kpi_d[1]]],
                  colWidths=[3.2*cm]*6)
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),L),("GRID",(0,0),(-1,-1),0.3,MI),
        ("ROWHEIGHT",(0,0),(-1,0),1.0*cm),("ALIGN",(0,0),(-1,-1),"CENTER"),
    ]))

    story += [Paragraph("📊 Executive Summary", H1), HRFlowable(width="100%",thickness=1,color=MI),
              Spacer(1,0.3*cm), kpi_t, Spacer(1,0.6*cm)]

    findings = [
        f"• Spans {(df['Date'].max()-df['Date'].min()).days} days, {df['ATM_ID'].nunique()} ATMs, {df['Location_Type'].nunique()} location types.",
        f"• Best model: {art['best']['name']} — R²={r2:.4f}, RMSE=₹{rmse:,.0f}.",
        f"• Holiday days drive {(hol_n/nrm_n-1)*100:+.1f}% higher average withdrawals.",
        f"• {na} anomalies detected across the full dataset ({na/len(df)*100:.1f}%).",
        f"• {int(crit)} CRITICAL cash shortage alerts requiring immediate action.",
    ]
    story += [Paragraph("Key Findings", H2)]
    for f in findings:
        story += [Paragraph(f, BD), Spacer(1,0.1*cm)]
    story.append(PageBreak())

    # EDA
    story.append(Paragraph("📈 Exploratory Data Analysis", H1))
    story.append(HRFlowable(width="100%",thickness=1,color=MI))
    for key, title, ins in [
        ("distributions",   "1. Cash Flow Distributions",
         "Right-skewed withdrawal distribution — tail demand events require buffer stock above mean forecasts."),
        ("time_series",     "2. Time Series",
         "Gradual upward trend in 2023. Month-start salary spikes and festive-season surges clearly visible."),
        ("weekday_weekend", "3. Weekday vs Weekend",
         "Weekends show higher median withdrawals. Stock ATMs on Thursday evening for weekend demand."),
        ("boxplots",        "4. Outlier Detection",
         "Mall and Bank Branch ATMs have the widest IQR. Snowy weather suppresses footfall significantly."),
        ("correlation",     "5. Feature Correlations",
         "Rolling averages and lag features are the strongest predictors — confirming autocorrelation."),
        ("holiday_impact",  "6. Holiday & Event Impact",
         "Holidays and events drive measurable uplift. High error bars demand conservative buffer stocking."),
        ("external",        "7. Weather & Competition",
         "Clear weather maximises ATM usage. More competitors reduce per-machine demand."),
        ("heatmap",         "8. Demand Heatmap",
         "Mall/Friday-Saturday cells carry peak demand. Gas Station/Sunday consistently underutilised."),
    ]:
        story.append(KeepTogether([
            Paragraph(title, H2), img(eda_paths.get(key)),
            Paragraph(f"💡 {ins}", INS), Spacer(1,0.5*cm)
        ]))
    story.append(PageBreak())

    # Clustering
    story.append(Paragraph("🔵 Clustering Analysis", H1))
    story.append(HRFlowable(width="100%",thickness=1,color=MI))
    for key, title in [("elbow","Elbow + Silhouette"),
                        ("scatter","PCA Cluster Scatter"),
                        ("comparison","Cluster Comparison")]:
        story.append(Paragraph(title, H2))
        story.append(img(cluster_paths.get(key))); story.append(Spacer(1,0.3*cm))
    story.append(PageBreak())

    # Anomaly
    story.append(Paragraph("🚨 Anomaly Detection", H1))
    story.append(HRFlowable(width="100%",thickness=1,color=MI))
    for key, title in [("timeseries","Anomaly Time Series"),
                        ("overview","Method Comparison"),
                        ("context","Anomaly Context")]:
        p_ = anomaly_paths.get(key)
        if p_:
            story.append(Paragraph(title, H2))
            story.append(img(p_)); story.append(Spacer(1,0.3*cm))
    story.append(PageBreak())

    # Model results
    story.append(Paragraph("🤖 Prediction Model Results", H1))
    story.append(HRFlowable(width="100%",thickness=1,color=MI))
    story.append(img(os.path.join(OUT_ROOT,"models","16_actual_vs_predicted.png")))
    story.append(img(os.path.join(OUT_ROOT,"models","15_feature_importance.png")))
    mdata = [["Metric","Value","Interpretation"],
             ["RMSE",    f"₹{m['RMSE']:,.0f}",  "Average prediction error"],
             ["MAE",     f"₹{m['MAE']:,.0f}",   "Median absolute error"],
             ["R² Score",f"{m['R2']:.4f}",       f"Explains {m['R2']*100:.1f}% of variance"]]
    story.append(tbl_style(mdata, [4*cm,5*cm,9*cm]))
    story.append(PageBreak())

    # Alerts
    story.append(Paragraph("🔔 Alert System & Cash Decisions", H1))
    story.append(HRFlowable(width="100%",thickness=1,color=MI))
    story.append(img(alert_art.get("dashboard_path"), h=10*cm))
    rt = alert_art.get("report_table")
    if rt is not None and len(rt):
        cols = [c for c in ["ATM_ID","Location_Type","Alert_Type","Recommended_Refill"] if c in rt.columns]
        data = [cols] + rt[cols].head(15).astype(str).values.tolist()
        story.append(Paragraph("Top Priority ATMs", H2))
        story.append(tbl_style(data))
    story.append(PageBreak())
    story.append(Spacer(1,5*cm))
    story.append(Paragraph("End of Report", sty("end",fontSize=14,textColor=colors.grey,alignment=TA_CENTER)))

    doc = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=1.8*cm, rightMargin=1.8*cm,
                             topMargin=1.5*cm, bottomMargin=1.5*cm)
    doc.build(story)
    print(f"[REPORT] PDF → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_full_pipeline(data_path=DATA_PATH):
    t0 = time.time()
    _banner("ATM CASH DEMAND FORECASTING — FULL PIPELINE")
    _mkdir(OUT_ROOT)

    df, pre_art              = preprocess(data_path)
    df                       = build_features(df)
    eda_paths                = run_eda(df)
    df, cluster_art          = run_clustering(df)
    df, anomaly_art          = run_anomaly_detection(df)
    df, model_art, mdl_paths = run_modeling(df)
    df, alert_art            = run_alerts(df)

    # Export enriched CSV
    print("\n" + "="*60 + "\n  STEP 8 — EXPORT\n" + "="*60)
    csv_path = os.path.join(OUT_ROOT, "atm_enriched_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"[EXPORT] CSV → {csv_path}  |  Shape: {df.shape}")

    # PDF report
    print("\n" + "="*60 + "\n  STEP 9 — PDF REPORT\n" + "="*60)
    try:
        pdf_path = generate_pdf_report(
            df, model_art, eda_paths,
            cluster_art["paths"], anomaly_art["paths"], alert_art
        )
    except Exception as e:
        print(f"[WARN] PDF failed: {e}"); pdf_path = None

    # Final summary
    elapsed = time.time() - t0
    _banner("PIPELINE COMPLETE ✓")
    m = model_art["best"]["metrics"]
    print(f"  Records          : {len(df):,}")
    print(f"  ATMs             : {df['ATM_ID'].nunique()}")
    print(f"  Features Created : {df.shape[1]}")
    print(f"  Clusters         : {df['Cluster_Label'].nunique()}")
    print(f"  Anomalies        : {df['Is_Anomaly'].sum():,} ({df['Is_Anomaly'].mean()*100:.1f}%)")
    print(f"  Best Model       : {model_art['best']['name']}")
    print(f"    R²   = {m['R2']:.4f}")
    print(f"    RMSE = ₹{m['RMSE']:,.0f}")
    print(f"    MAE  = ₹{m['MAE']:,.0f}")
    sev = df["Alert_Severity"].value_counts()
    for s,e in [("CRITICAL","🔴"),("HIGH","🟠"),("MEDIUM","🟡"),("LOW","🟢"),("NORMAL","✅")]:
        n = sev.get(s,0); bar = "█"*(n*25//max(sev.values))
        print(f"  Alert {s:9s}: {n:>5,}  {e} {bar}")
    print(f"\n  Output folder    : {OUT_ROOT}")
    print(f"  CSV exported     : {csv_path}")
    if pdf_path: print(f"  PDF report       : {pdf_path}")
    print(f"  Total time       : {elapsed:.1f}s\n")
    print("  Run dashboard:  streamlit run atm_system.py\n")

    return df, {"model": model_art, "eda": eda_paths,
                "clusters": cluster_art, "anomaly": anomaly_art,
                "alerts": alert_art, "csv": csv_path, "pdf": pdf_path}


# ═══════════════════════════════════════════════════════════════════════
#  STREAMLIT INTERACTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="ATM Cash Intelligence",
                       page_icon="🏧", layout="wide",
                       initial_sidebar_state="expanded")

    # ── CSS ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    .main-hdr{background:linear-gradient(135deg,#1A1A2E,#16213E);padding:1.2rem 2rem;
               border-radius:12px;margin-bottom:1.5rem;}
    .insight{background:#E8F5E9;border-left:4px solid #2D6A4F;padding:0.5rem 1rem;
              border-radius:6px;font-style:italic;margin:0.4rem 0;}
    .kpi-card{background:#F8F9FA;border-left:4px solid #E63946;
               padding:0.8rem 1rem;border-radius:8px;margin:0.2rem 0;}
    </style>""", unsafe_allow_html=True)

    # ── CACHED DATA LOAD ─────────────────────────────────────
    @st.cache_data(show_spinner="Running full pipeline… (~30s)")
    def _load():
        df, _ = preprocess(DATA_PATH)
        df    = build_features(df)
        df    = detect_zscore(df); df = detect_iqr(df)
        df, _ = detect_isoforest(df); df = build_consensus(df)
        Xsc, _= _prepare_cluster_data(df)
        km, lbl = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42).fit_predict(Xsc).__class__, None
        km    = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
        lbl   = km.fit_predict(Xsc)
        df    = _assign_cluster_labels(df, lbl)
        art   = train_models(df)
        df    = annotate_predictions(df, art)
        df    = generate_alerts(df)
        return df, art
    df_full, model_art = _load()

    # ── SIDEBAR FILTERS ──────────────────────────────────────
    st.sidebar.markdown("## 🔧 Filters")
    st.sidebar.markdown("---")
    min_d = df_full["Date"].min().date(); max_d = df_full["Date"].max().date()
    dr = st.sidebar.date_input("📅 Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
    d0 = pd.Timestamp(dr[0]) if len(dr)>0 else pd.Timestamp(min_d)
    d1 = pd.Timestamp(dr[1]) if len(dr)>1 else pd.Timestamp(max_d)

    days_all = sorted(df_full["Day_of_Week"].dropna().unique())
    locs_all = sorted(df_full["Location_Type"].dropna().unique())
    times_all= sorted(df_full["Time_of_Day"].dropna().unique())
    clus_all = sorted(df_full["Cluster_Label"].dropna().unique())

    sel_days  = st.sidebar.multiselect("📆 Day of Week",   days_all,  default=days_all)
    sel_locs  = st.sidebar.multiselect("📍 Location Type", locs_all,  default=locs_all)
    sel_times = st.sidebar.multiselect("⏰ Time of Day",   times_all, default=times_all)
    sel_clus  = st.sidebar.multiselect("🔵 Cluster",       clus_all,  default=clus_all)
    sel_sev   = st.sidebar.multiselect("🚦 Alert Severity",
                                        ["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"],
                                        default=["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"])
    only_anoms= st.sidebar.checkbox("Anomaly Records Only", False)

    df = df_full.copy()
    df = df[(df["Date"]>=d0) & (df["Date"]<=d1)]
    if sel_days:  df = df[df["Day_of_Week"].isin(sel_days)]
    if sel_locs:  df = df[df["Location_Type"].isin(sel_locs)]
    if sel_times: df = df[df["Time_of_Day"].isin(sel_times)]
    if sel_clus:  df = df[df["Cluster_Label"].isin(sel_clus)]
    if sel_sev:   df = df[df["Alert_Severity"].isin(sel_sev)]
    if only_anoms:df = df[df["Is_Anomaly"]==1]

    # ── HEADER ───────────────────────────────────────────────
    st.markdown("""<div class="main-hdr">
      <h1 style="margin:0;font-size:1.9rem;color:white;">🏧 ATM Cash Demand Intelligence System</h1>
      <p style="margin:0.3rem 0 0;color:#A8DADC;font-size:0.9rem;">
        End-to-End Forecasting · Clustering · Anomaly Detection · Decision Alerts
      </p></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📋 Records",        f"{len(df):,}")
    c2.metric("💸 Avg Withdrawal",  f"₹{df['Total_Withdrawals'].mean()/1e3:.1f}K" if len(df) else "—")
    c3.metric("🚨 Anomalies",       int(df["Is_Anomaly"].sum()))
    c4.metric("🔴 Critical Alerts", int((df["Alert_Severity"]=="CRITICAL").sum()))
    c5.metric("🎯 Avg Confidence",  f"{df['Confidence_Score'].mean()*100:.1f}%" if "Confidence_Score" in df.columns else "—")
    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────
    t1,t2,t3,t4,t5 = st.tabs(["📊 EDA","🔵 Clusters","🚨 Anomalies","🤖 Predictions","🔔 Alerts"])

    # ── TAB 1: EDA ───────────────────────────────────────────
    with t1:
        st.subheader("📊 Exploratory Data Analysis")
        col1,col2 = st.columns(2)

        # Distribution
        with col1:
            st.markdown("**Withdrawal Distribution**")
            fig,ax = plt.subplots(figsize=(6,4))
            d = df["Total_Withdrawals"].dropna()
            ax.hist(d,bins=40,color=C_RED,alpha=0.7,edgecolor="white",density=True)
            if len(d)>1: d.plot.kde(ax=ax,color="black",lw=2)
            ax.axvline(d.mean(),color="gold",lw=2,ls="--",label=f"Mean ₹{d.mean()/1e3:.1f}K")
            ax.axvline(d.median(),color="lime",lw=2,ls=":",label=f"Med ₹{d.median()/1e3:.1f}K")
            ax.xaxis.set_major_formatter(FMT); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">💡 Right-skewed → tail demand events require buffer stocking above mean.</div>',
                        unsafe_allow_html=True)

        with col2:
            st.markdown("**Next-Day Demand Distribution**")
            fig,ax = plt.subplots(figsize=(6,4))
            d2 = df["Cash_Demand_Next_Day"].dropna()
            ax.hist(d2,bins=40,color=C_BLUE,alpha=0.7,edgecolor="white",density=True)
            if len(d2)>1: d2.plot.kde(ax=ax,color="black",lw=2)
            ax.axvline(d2.mean(),color="gold",lw=2,ls="--",label=f"Mean ₹{d2.mean()/1e3:.1f}K")
            ax.xaxis.set_major_formatter(FMT); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**📈 Time Series — Daily Average Withdrawals**")
        daily = df.groupby("Date")[["Total_Withdrawals","Cash_Demand_Next_Day"]].mean().reset_index()
        fig,ax = plt.subplots(figsize=(13,4))
        ax.plot(daily["Date"],daily["Total_Withdrawals"],color=C_RED,lw=1.1,alpha=0.8,label="Withdrawals")
        ax.plot(daily["Date"],daily["Cash_Demand_Next_Day"],color=C_BLUE,lw=1.1,alpha=0.8,label="Next-Day Demand")
        ax.plot(daily["Date"],daily["Total_Withdrawals"].rolling(30,min_periods=1).mean(),
                color="black",lw=2,ls="--",label="30-day MA")
        ax.yaxis.set_major_formatter(FMT); ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        col3,col4 = st.columns(2)
        with col3:
            st.markdown("**Day of Week Analysis**")
            dm = df[df["Day_of_Week"].isin(DAY_ORDER)].groupby("Day_of_Week")["Total_Withdrawals"].mean().reindex(DAY_ORDER).dropna()
            fig,ax = plt.subplots(figsize=(6,4))
            ax.bar(dm.index,dm.values,color=[C_RED if d in ["Saturday","Sunday"] else C_BLUE for d in dm.index],edgecolor="white")
            ax.yaxis.set_major_formatter(FMT); ax.tick_params(axis="x",rotation=30)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col4:
            st.markdown("**Holiday vs Normal**")
            hm = df.groupby("Holiday_Flag")["Total_Withdrawals"].mean()
            fig,ax = plt.subplots(figsize=(6,4))
            lbl = ["Normal","Holiday"] if len(hm)==2 else hm.index.tolist()
            ax.bar(lbl[:len(hm)],hm.values,color=[C_BLUE,C_RED][:len(hm)],edgecolor="white")
            ax.yaxis.set_major_formatter(FMT)
            for i,v in enumerate(hm.values):
                ax.text(i,v+200,f"₹{v/1e3:.1f}K",ha="center",fontsize=9,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        col5,col6 = st.columns(2)
        with col5:
            st.markdown("**Weather Impact**")
            wm = df.groupby("Weather_Condition")["Total_Withdrawals"].mean().sort_values(ascending=True)
            fig,ax = plt.subplots(figsize=(6,4))
            ax.barh(wm.index,wm.values,color=C_BLUE,edgecolor="white")
            ax.xaxis.set_major_formatter(FMT)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col6:
            st.markdown("**Demand Heatmap (Location × Day)**")
            if len(df)>10:
                piv = df.pivot_table(values="Cash_Demand_Next_Day",index="Location_Type",
                                      columns="Day_of_Week",aggfunc="mean")
                piv = piv.reindex(columns=[d for d in DAY_ORDER if d in piv.columns])
                fig,ax = plt.subplots(figsize=(6,4))
                sns.heatmap(piv,annot=True,fmt=".0f",cmap="YlOrRd",ax=ax,
                            linewidths=0.3,annot_kws={"size":7})
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TAB 2: CLUSTERS ──────────────────────────────────────
    with t2:
        st.subheader("🔵 ATM Demand Clusters")
        cc = df["Cluster_Label"].value_counts()
        c1,c2,c3 = st.columns(3)
        for col,lbl in zip([c1,c2,c3],["High Demand","Medium Demand","Low Demand"]):
            n = cc.get(lbl,0)
            col.metric(lbl, f"{n:,}", delta=f"{n/max(len(df),1)*100:.1f}%")

        col7,col8 = st.columns(2)
        with col7:
            st.markdown("**Cluster Size**")
            vals = [cc.get(l,0) for l in ["High Demand","Medium Demand","Low Demand"]]
            fig,ax = plt.subplots(figsize=(6,4))
            ax.pie(vals,labels=["High","Medium","Low"],
                   colors=[C_RED,C_ORG,C_BLUE],autopct="%1.1f%%",startangle=140)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col8:
            st.markdown("**Avg Withdrawal per Cluster**")
            cm_w = df.groupby("Cluster_Label")["Total_Withdrawals"].mean()
            fig,ax = plt.subplots(figsize=(6,4))
            ax.bar(cm_w.index,cm_w.values,
                   color=[CLUSTER_COLORS.get(l,"#888") for l in cm_w.index],edgecolor="white")
            ax.yaxis.set_major_formatter(FMT); ax.tick_params(axis="x",rotation=15)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**Cluster Characteristics**")
        metrics_c = [c for c in ["Total_Withdrawals","Cash_Utilization_Rate",
                                   "Urban_Demand_Index","Demand_Volatility"] if c in df.columns]
        if metrics_c:
            st.dataframe(df.groupby("Cluster_Label")[metrics_c].mean().round(3)
                           .style.background_gradient(cmap="RdYlGn",axis=0),
                         use_container_width=True)

        for lbl,em,desc in [
            ("High Demand","🔴","Urban malls & bank branches. High withdrawal volume & cash utilization. Needs frequent replenishment."),
            ("Medium Demand","🟠","Supermarkets & standalone ATMs. Moderate predictable demand. Weekly schedule sufficient."),
            ("Low Demand","🔵","Gas stations & rural ATMs. Low footfall. Risk of excess idle cash."),
        ]:
            st.markdown(f"**{em} {lbl}:** {desc}")

    # ── TAB 3: ANOMALIES ─────────────────────────────────────
    with t3:
        st.subheader("🚨 Anomaly Detection")
        ca1,ca2,ca3 = st.columns(3)
        ca1.metric("Z-Score",          int(df.get("Is_Anomaly_ZScore",   pd.Series(dtype=int)).sum()))
        ca2.metric("IQR",              int(df.get("Is_Anomaly_IQR",      pd.Series(dtype=int)).sum()))
        ca3.metric("Isolation Forest", int(df.get("Is_Anomaly_IsoForest",pd.Series(dtype=int)).sum()))

        st.markdown("---")
        daily_a = df.groupby("Date").agg(W=("Total_Withdrawals","mean"),A=("Is_Anomaly","max")).reset_index()
        fig,ax  = plt.subplots(figsize=(13,4))
        ax.plot(daily_a["Date"],daily_a["W"],color=C_BLUE,lw=1.1,alpha=0.8,label="Daily Avg")
        ax.plot(daily_a["Date"],daily_a["W"].rolling(14,min_periods=1).mean(),
                color="black",lw=2,ls="--",alpha=0.7,label="14-day MA")
        an = daily_a[daily_a["A"]==1]
        ax.scatter(an["Date"],an["W"],color=C_RED,s=55,zorder=5,label="Anomaly")
        ax.yaxis.set_major_formatter(FMT); ax.legend(fontsize=9)
        ax.set_title("Anomaly Time Series"); plt.tight_layout(); st.pyplot(fig); plt.close()

        anom_df = df[df["Is_Anomaly"]==1]
        if len(anom_df):
            cols_a = [c for c in ["Date","ATM_ID","Location_Type","Total_Withdrawals",
                                   "Zscore_Withdrawal","Anomaly_Votes","Holiday_Flag"] if c in anom_df.columns]
            st.dataframe(anom_df[cols_a].sort_values("Total_Withdrawals",ascending=False).head(50),
                         use_container_width=True)
        else:
            st.info("No anomalies in current filter.")

        st.markdown('<div class="insight">💡 Consensus flag (≥2 methods) minimises false positives. '
                    'Most anomalies align with holidays and special events — expected demand spikes.</div>',
                    unsafe_allow_html=True)

    # ── TAB 4: PREDICTIONS ───────────────────────────────────
    with t4:
        st.subheader("🤖 Cash Demand Predictions")
        m  = model_art["best"]["metrics"]
        nm = model_art["best"]["name"]
        pm1,pm2,pm3,pm4 = st.columns(4)
        pm1.metric("Best Model", nm)
        pm2.metric("R² Score",   f"{m['R2']:.4f}")
        pm3.metric("RMSE",       f"₹{m['RMSE']/1e3:.1f}K")
        pm4.metric("MAE",        f"₹{m['MAE']/1e3:.1f}K")

        col_p1,col_p2 = st.columns(2)
        with col_p1:
            y_te = model_art["data"]["y_te"]; preds = m["predictions"]
            fig,ax = plt.subplots(figsize=(6,5))
            ax.scatter(y_te,preds,alpha=0.35,s=14,color=C_BLUE,edgecolors="none")
            lo=min(y_te.min(),preds.min()); hi=max(y_te.max(),preds.max())
            ax.plot([lo,hi],[lo,hi],"r--",lw=2,label="Perfect")
            ax.xaxis.set_major_formatter(FMT); ax.yaxis.set_major_formatter(FMT)
            ax.set_title(f"Actual vs Predicted (R²={m['R2']:.4f})"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_p2:
            imp = pd.Series(model_art["best"]["model"].feature_importances_,
                            index=model_art["features"]).sort_values(ascending=False).head(15)
            fig,ax = plt.subplots(figsize=(6,5))
            clrs = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(imp)))[::-1]
            ax.barh(imp.index[::-1],imp.values[::-1],color=clrs,edgecolor="white")
            ax.set_xlabel("Importance"); ax.set_title("Top 15 Feature Importances")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Real-time prediction form
        st.markdown("---")
        st.markdown("### 🔮 Real-Time Cash Demand Predictor")
        with st.form("rtpf"):
            r1,r2,r3 = st.columns(3)
            iw  = r1.number_input("💸 Total Withdrawals (₹)", 5000,150000,50000,1000)
            id_ = r2.number_input("💰 Total Deposits (₹)",       0,100000,10000,1000)
            ic  = r3.number_input("🏧 Prev Cash Level (₹)",      0,500000,100000,5000)
            r4,r5,r6 = st.columns(3)
            ih  = r4.selectbox("🎉 Holiday?",       [0,1],format_func=lambda x:"Yes" if x else "No")
            ie  = r5.selectbox("🎊 Special Event?", [0,1],format_func=lambda x:"Yes" if x else "No")
            ik  = r6.number_input("🏪 Competitor ATMs",0,10,2)
            r7,r8,r9 = st.columns(3)
            il  = r7.selectbox("📍 Location Type", locs_all)
            iday= r8.selectbox("📅 Day of Week",   DAY_ORDER)
            it  = r9.selectbox("⏰ Time of Day",   times_all)
            go  = st.form_submit_button("🚀 Predict Now")

        if go:
            import datetime as _dt
            today = _dt.date.today()
            sm = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
            uw = {"Mall":1.0,"Bank Branch":0.9,"Supermarket":0.7,"Standalone":0.5,"Gas Station":0.4}
            row = {
                "Total_Withdrawals":iw,"Total_Deposits":id_,"Previous_Day_Cash_Level":ic,
                "Is_Weekend":int(iday in ["Saturday","Sunday"]),
                "Month":today.month,"Quarter":(today.month-1)//3+1,
                "Day_of_Year":today.timetuple().tm_yday,"Year":today.year,
                "Season":sm[today.month],"Is_Month_Start":int(today.day<=3),
                "Is_Month_End":int(today.day>=28),"Day_of_Month":today.day,
                "Rolling_3Day_Withdrawals":iw,"Rolling_7Day_Withdrawals":iw,
                "Rolling_3Day_Deposits":id_,"Rolling_7Day_Deposits":id_,
                "Rolling_7Day_Std":0,"Withdrawal_Growth_Rate":0,
                "Deposit_Growth_Rate":0,"Demand_Trend_Slope":0,
                "Withdrawal_Deposit_Ratio":iw/(id_+1),
                "Cash_Utilization_Rate":min(iw/(ic+1),5),
                "Demand_to_Cash_Ratio":min(iw/(ic+1),5),
                "Excess_Cash_Level":max(0,ic-iw),
                "Holiday_Flag":ih,"Special_Event_Flag":ie,
                "Holiday_Impact_Score":ih*1.5,"Event_Impact_Score":ie*1.25,
                "Days_Since_Last_Holiday":0,"Days_Until_Next_Event":0,
                "Nearby_Competitor_ATMs":ik,"Competitor_Density_Level":min(2,ik//2),
                "Urban_Demand_Index":uw.get(il,0.5),"Location_Risk_Score":0.5,
                "Withdrawal_Lag_1":iw,"Withdrawal_Lag_2":iw,"Withdrawal_Lag_7":iw,
                "Demand_Volatility":0.1,"Peak_Hour_Flag":int(it in ["Evening","Morning"]),
                "Low_Activity_Flag":0,"Is_Anomaly":0,"Anomaly_Votes":0,"Zscore_Withdrawal":0,
                "ATM_ID_Enc":0,
                "Day_of_Week_Enc":DAY_ORDER.index(iday) if iday in DAY_ORDER else 0,
                "Time_of_Day_Enc":times_all.index(it) if it in times_all else 0,
                "Weather_Condition_Enc":0,
                "Location_Type_Bank Branch":int(il=="Bank Branch"),
                "Location_Type_Gas Station":int(il=="Gas Station"),
                "Location_Type_Mall":int(il=="Mall"),
                "Location_Type_Standalone":int(il=="Standalone"),
                "Location_Type_Supermarket":int(il=="Supermarket"),
            }
            feats = model_art["features"]
            Xin   = pd.DataFrame([{f:row.get(f,0) for f in feats}])
            pred_v= int(model_art["best"]["model"].predict(Xin)[0])
            refill= max(0, int(pred_v*1.20 - ic))
            short = ic < pred_v * 0.80

            pr1,pr2,pr3 = st.columns(3)
            pr1.metric("🔮 Predicted Demand",  f"₹{pred_v:,}")
            pr2.metric("💰 Recommended Refill", f"₹{refill:,}")
            pr3.metric("⚠️ Shortage Risk",       "HIGH ⚠️" if short else "LOW ✅",
                       delta_color="inverse")
            if short:
                st.error(f"🔴 CRITICAL: Cash ₹{ic:,} < 80% of predicted demand ₹{pred_v:,}. "
                         f"Refill ₹{refill:,} immediately.")
            elif ih or ie:
                st.warning("🟡 Holiday/Event detected. Consider +25% buffer on top of recommendation.")
            else:
                st.success(f"✅ Cash level adequate. Recommended top-up: ₹{refill:,}")

        # Sample predictions table
        st.markdown("---")
        st.markdown("**Sample Predictions**")
        pcols = [c for c in ["Date","ATM_ID","Location_Type","Total_Withdrawals",
                              "Cash_Demand_Next_Day","Predicted_Demand",
                              "Prediction_Error","Confidence_Score"] if c in df.columns]
        st.dataframe(df[pcols].head(100), use_container_width=True)

    # ── TAB 5: ALERTS ────────────────────────────────────────
    with t5:
        st.subheader("🔔 Alert System & Cash Decisions")
        sev = df["Alert_Severity"].value_counts()
        a1,a2,a3,a4,a5 = st.columns(5)
        for col,s,e in zip([a1,a2,a3,a4,a5],
                ["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"],
                ["🔴","🟠","🟡","🟢","✅"]):
            col.metric(f"{e} {s}", f"{sev.get(s,0):,}")

        col_a1,col_a2 = st.columns(2)
        with col_a1:
            sc2 = sev.reindex(["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"]).dropna()
            fig,ax = plt.subplots(figsize=(5,4))
            ax.pie(sc2.values,labels=sc2.index,
                   colors=[C_RED,C_ORG,"#FFBA08",C_GREEN,"#4CAF50"][:len(sc2)],
                   autopct="%1.1f%%",startangle=140,textprops={"fontsize":8})
            ax.set_title("Alert Severity Mix"); plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_a2:
            lr2 = df.groupby("Location_Type")["Recommended_Refill"].mean().sort_values(ascending=True)
            fig,ax = plt.subplots(figsize=(5,4))
            ax.barh(lr2.index,lr2.values,color=C_RED,edgecolor="white")
            ax.xaxis.set_major_formatter(FMT)
            ax.set_title("Avg Refill by Location"); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**🏆 Priority ATMs — Action Required**")
        rt = priority_atm_table(df, n=25)
        if len(rt):
            sev_bg = {"🔴 CRITICAL: Cash Shortage":"#FFE5E5",
                      "🟠 HIGH: Low Cash Warning":"#FFF0E0",
                      "🟡 MEDIUM: Holiday Spike":"#FFFBE5",
                      "🟡 MEDIUM: Unusual Activity":"#FFFBE5"}
            def hl(val): return f"background-color:{sev_bg.get(val,'')}"
            at_col = "Alert_Type" if "Alert_Type" in rt.columns else rt.columns[-1]
            st.dataframe(rt.style.applymap(hl, subset=[at_col]), use_container_width=True)
        else:
            st.info("No alerts in current filter selection.")

        st.markdown("---")
        st.markdown("### ⬇️ Download Reports")
        dc1,dc2,dc3 = st.columns(3)
        with dc1:
            st.download_button("📥 Full Dataset CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               "atm_full_data.csv","text/csv",
                               use_container_width=True)
        with dc2:
            if len(rt):
                st.download_button("🚨 Alert Report CSV",
                                   rt.to_csv(index=False).encode("utf-8"),
                                   "atm_alerts.csv","text/csv",
                                   use_container_width=True)
        with dc3:
            an2 = df[df["Is_Anomaly"]==1]
            if len(an2):
                st.download_button("🔍 Anomaly Records CSV",
                                   an2.to_csv(index=False).encode("utf-8"),
                                   "atm_anomalies.csv","text/csv",
                                   use_container_width=True)

        # Sidebar model info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Model Info")
        st.sidebar.markdown(f"**Best:** {model_art['best']['name']}")
        st.sidebar.markdown(f"**R²:** {model_art['best']['metrics']['R2']:.4f}")
        st.sidebar.markdown(f"**RMSE:** ₹{model_art['best']['metrics']['RMSE']:,.0f}")
        st.sidebar.markdown("---")
        st.sidebar.caption("ATM Cash Intelligence System v2.0")


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Detect if called via `streamlit run`
    try:
        import streamlit as _st
        if _st.runtime.exists():
            streamlit_app()
        else:
            run_full_pipeline()
    except Exception:
        run_full_pipeline()
