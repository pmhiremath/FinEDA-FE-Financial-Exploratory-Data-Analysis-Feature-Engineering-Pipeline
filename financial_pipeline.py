"""
  Author  : [Prajwal kapnoor]
  Dataset : S&P 500 Stock Data (Kaggle) — historical OHLCV prices
  Stack   : Python | Pandas | NumPy | Scikit-Learn | Matplotlib | Seaborn
  Purpose : End-to-end pipeline from raw CSV to a model-ready feature matrix
=============================================================================
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Core Data ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── Visualisation ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ─── Scikit-Learn ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# ─── Aesthetics ───────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "figure.dpi":       150,
})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — Synthetic Dataset Generator
#  WHY: Kaggle's S&P 500 CSV mirrors this schema exactly (Date, Open, High,
#       Low, Close, Volume, Name). Generating data here keeps the notebook
#       self-contained for reviewers who don't have the CSV handy, while
#       demonstrating that you understand the real data structure.
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    tickers: list[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate multi-ticker OHLCV data that replicates the Kaggle
    'S&P 500 Stock Data' schema, including realistic messiness:
      - ~3 % missing values (NaN) scattered across price columns
      - ~1 % outlier spikes (fat-finger / data-feed errors)
      - A deliberate wrong dtype on the Date column (object/string)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")          # business days only
    records = []

    for ticker in tickers:
        n = len(dates)
        # Geometric Brownian Motion for Close price (realistic drift + vol)
        S0     = rng.uniform(50, 500)                    # random starting price
        mu     = rng.uniform(0.0001, 0.0003)             # daily drift
        sigma  = rng.uniform(0.01, 0.025)                # daily volatility
        shocks = rng.standard_normal(n)
        close  = S0 * np.cumprod(np.exp((mu - 0.5 * sigma**2) + sigma * shocks))

        # OHLCV derived from Close
        spread  = rng.uniform(0.005, 0.015, n)
        high    = close * (1 + spread)
        low     = close * (1 - spread)
        open_   = close * (1 + rng.uniform(-0.005, 0.005, n))
        volume  = rng.integers(1_000_000, 50_000_000, n).astype(float)

        df_t = pd.DataFrame({
            "Date":   dates.astype(str),           # ← intentionally wrong dtype
            "Open":   np.round(open_,   2),
            "High":   np.round(high,    2),
            "Low":    np.round(low,     2),
            "Close":  np.round(close,   2),
            "Volume": volume,
            "Name":   ticker,
        })

        # ── Inject ~3 % NaNs ──────────────────────────────────────────────
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            mask = rng.random(n) < 0.03
            df_t.loc[mask, col] = np.nan

        # ── Inject ~1 % price outliers (fat-finger spikes) ────────────────
        for col in ["Close", "High"]:
            spike_idx = rng.choice(n, size=int(n * 0.01), replace=False)
            df_t.loc[spike_idx, col] *= rng.uniform(1.5, 3.0, len(spike_idx))

        records.append(df_t)

    raw = pd.concat(records, ignore_index=True)
    print(f"[Dataset Generated]  Shape: {raw.shape}  |  Tickers: {tickers}")
    return raw


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA EXTRACTION & LOADING
#  WHY: Real Kaggle CSVs arrive in a wide multi-ticker format. Pinning
#       usecols limits memory overhead — critical when the file is >1 GB.
#       Converting Date immediately prevents silent string-comparison bugs
#       that corrupt all downstream time-series operations.
# ══════════════════════════════════════════════════════════════════════════════

ESSENTIAL_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Name"]

def load_and_isolate(
    raw_df: pd.DataFrame,
    ticker: str = "AAPL",
) -> pd.DataFrame:
    """
    Simulate loading the Kaggle CSV (usecols pin) then isolate one ticker.
    In production replace `raw_df` with:
        pd.read_csv("all_stocks_5yr.csv", usecols=ESSENTIAL_COLUMNS,
                    parse_dates=["Date"])
    """
    print("\n" + "═" * 60)
    print("  STEP 1 — DATA EXTRACTION & LOADING")
    print("═" * 60)

    # ── Column isolation (mirrors usecols on pd.read_csv) ─────────────────
    df = raw_df[ESSENTIAL_COLUMNS].copy()
    print(f"\n[1.1]  Raw shape after column isolation : {df.shape}")

    # ── Enforce datetime — catches the 'object' dtype injected above ───────
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Filter to a single ticker for a focused single-asset pipeline ──────
    df = df[df["Name"] == ticker].copy()
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)            # DatetimeIndex for .resample()
    df.drop(columns=["Name"], inplace=True)

    print(f"[1.2]  Ticker '{ticker}' isolated      : {df.shape}")
    print(f"[1.3]  Index dtype                     : {df.index.dtype}")
    print(f"\n[1.4]  First 5 rows:\n{df.head()}")
    print(f"\n[1.5]  DTYPEs:\n{df.dtypes}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — EXPLORATORY DATA ANALYSIS
#  WHY: In quant finance, "look before you leap" is risk management.
#       Skewness in returns signals fat tails (crash risk). Correlation
#       between OHLC columns often exceeds 0.99 — a multicollinearity
#       red-flag that EDA catches before it silently inflates model variance.
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    print("═" * 60)

    # ── 2.1 Summary statistics ─────────────────────────────────────────────
    print("\n[2.1]  Descriptive Statistics:")
    desc = df.describe().T
    desc["skewness"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    print(desc.round(3).to_string())

    # ── 2.2 Missing-value heatmap ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("EDA Dashboard — Financial Data Overview", fontsize=15,
                 color="#e0e0e0", fontweight="bold")

    missing_pct = df.isnull().mean() * 100
    axes[0].barh(missing_pct.index, missing_pct.values,
                 color=["#ef5350" if v > 0 else "#66bb6a" for v in missing_pct])
    axes[0].set_title("Missing Values (%)", color="#e0e0e0")
    axes[0].set_xlabel("% Missing")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#444")

    # ── 2.3 Close price distribution ──────────────────────────────────────
    clean_close = df["Close"].dropna()
    axes[1].hist(clean_close, bins=60, color="#42a5f5", edgecolor="#0d47a1",
                 alpha=0.85)
    axes[1].axvline(clean_close.mean(),  color="#ffca28", lw=2, ls="--",
                    label=f"Mean  {clean_close.mean():.2f}")
    axes[1].axvline(clean_close.median(), color="#ef9a9a", lw=2, ls=":",
                    label=f"Median {clean_close.median():.2f}")
    axes[1].set_title("Close Price Distribution", color="#e0e0e0")
    axes[1].legend(fontsize=8)

    # ── 2.4 Correlation heatmap ────────────────────────────────────────────
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=axes[2], annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#0f0f0f",
        annot_kws={"size": 9, "color": "white"},
    )
    axes[2].set_title("Feature Correlation Heatmap", color="#e0e0e0")
    axes[2].tick_params(colors="#e0e0e0")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/step2_eda_dashboard.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"\n[SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DATA CLEANING
#  WHY: Financial data has three endemic pathologies:
#    a) Missing values — exchange holidays, API timeouts, data-vendor gaps
#    b) Outliers — fat-finger errors, stock splits, erroneous ticks
#    c) Wrong dtypes — Date stored as string breaks .resample(), .shift(),
#       and every rolling window calculation downstream.
#  Production rule: impute prices with forward-fill (last known price);
#  clip outliers at IQR fence to preserve series continuity.
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  STEP 3 — DATA CLEANING")
    print("═" * 60)

    df = df.copy()

    # ── 3.1 Report missing values ─────────────────────────────────────────
    print("\n[3.1]  Missing values BEFORE cleaning:")
    print(df.isnull().sum().to_string())

    # ── 3.2 Forward-fill prices (last known price is best estimate) ────────
    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].ffill()

    # ── 3.3 Backward-fill any leading NaNs (series starts with NaN) ────────
    df[price_cols] = df[price_cols].bfill()

    # ── 3.4 Volume: fill with rolling median (volume is mean-reverting) ────
    df["Volume"] = df["Volume"].fillna(df["Volume"].rolling(10, min_periods=1).median())

    print("\n[3.2]  Missing values AFTER cleaning:")
    print(df.isnull().sum().to_string())

    # ── 3.5 Outlier clipping via IQR fence ────────────────────────────────
    #  IQR method is robust to fat-tailed distributions (unlike z-score).
    n_outliers_total = 0
    for col in price_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower  = Q1 - 3.0 * IQR          # 3× fence = conservative for finance
        upper  = Q3 + 3.0 * IQR
        n_out  = ((df[col] < lower) | (df[col] > upper)).sum()
        n_outliers_total += n_out
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[3.3]  {col:6s} outliers clipped: {n_out:4d}  "
              f"  fence=[{lower:.2f}, {upper:.2f}]")

    print(f"\n[3.4]  Total outliers clipped : {n_outliers_total}")

    # ── 3.6 Assert datetime index (contract for downstream steps) ─────────
    assert isinstance(df.index, pd.DatetimeIndex), \
        "FATAL: Index must be DatetimeIndex for time-series ops."
    assert df.isnull().sum().sum() == 0, "FATAL: Cleaning left residual NaNs."
    print("\n[3.5]  ✓ Assertions passed — DatetimeIndex confirmed, zero NaNs.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — DATA PREPROCESSING
#  WHY: ML algorithms (Ridge, SVM, KNN) are distance-based; unscaled Volume
#       (~10⁷) dominates Close (~10²) in any L2 norm. StandardScaler (μ=0,
#       σ=1) corrects this without distorting distributional shape.
#       Encoding Trend (Up/Down) as 0/1 makes it usable in linear models
#       and avoids the dummy-variable trap of one-hot encoding a binary col.
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    print("\n" + "═" * 60)
    print("  STEP 4 — DATA PREPROCESSING")
    print("═" * 60)

    df = df.copy()

    # ── 4.1 Volume log-transform (handles right-skew common in volume) ──────
    df["Log_Volume"] = np.log1p(df["Volume"])
    print("[4.1]  Log_Volume created  (skew fix for heavy-tailed Volume)")

    # ── 4.2 Numerical scaling ──────────────────────────────────────────────
    scale_cols = ["Open", "High", "Low", "Close", "Log_Volume"]
    scaler     = StandardScaler()
    scaled_arr = scaler.fit_transform(df[scale_cols])
    df_scaled  = pd.DataFrame(scaled_arr, columns=[f"{c}_scaled" for c in scale_cols],
                               index=df.index)
    df = pd.concat([df, df_scaled], axis=1)
    print(f"[4.2]  Scaled columns added : {list(df_scaled.columns)}")
    print(f"[4.3]  DataFrame shape after preprocessing : {df.shape}")

    return df, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — FINANCIAL FEATURE ENGINEERING
#  WHY: Raw OHLCV data is low-information. Feature engineering extracts the
#  *signal* that quants actually trade on:
#    • pct_change()   → Daily return, the fundamental unit of risk/reward
#    • rolling().mean()  → SMA filters noise; 20-day = ~1 trading month
#    • rolling().std()   → Realised volatility; underlies VaR calculations
#    • np.where()       → Vectorised conditional — O(n), not O(n²) like a loop
#  All operations are vectorised (Pandas C extensions under the hood) which
#  is 100-1000× faster than Python for-loops on time-series.
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  STEP 5 — FINANCIAL FEATURE ENGINEERING")
    print("═" * 60)

    df = df.copy()

    # ── 5.1 Daily Return (%) ───────────────────────────────────────────────
    df["Daily_Return_Pct"] = df["Close"].pct_change() * 100
    print("[5.1]  Daily_Return_Pct  — vectorised pct_change()")

    # ── 5.2 Log Return (continuous compounding, used in options pricing) ───
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    print("[5.2]  Log_Return        — ln(Pt / Pt-1)")

    # ── 5.3 Simple Moving Averages ─────────────────────────────────────────
    for window in [10, 20, 50]:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
    print("[5.3]  SMA_10, SMA_20, SMA_50 — rolling mean (trend filters)")

    # ── 5.4 Exponential Moving Average (reacts faster to recent price) ─────
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    print("[5.4]  EMA_20            — exponentially weighted mean")

    # ── 5.5 Rolling Volatility — 20-day annualised std of log returns ──────
    #  Annualisation: σ_annual ≈ σ_daily × √252  (trading days per year)
    df["Volatility_20D"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)
    print("[5.5]  Volatility_20D    — annualised realised volatility")

    # ── 5.6 Bollinger Bands ────────────────────────────────────────────────
    rolling_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * rolling_std
    df["BB_Lower"] = df["SMA_20"] - 2 * rolling_std
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"]
    print("[5.6]  BB_Upper, BB_Lower, BB_Width — Bollinger Band signals")

    # ── 5.7 Relative Strength Index (RSI — 14-day) ────────────────────────
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    print("[5.7]  RSI_14            — momentum oscillator (0–100)")

    # ── 5.8 MACD (Moving Average Convergence Divergence) ──────────────────
    ema_12         = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26         = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]     = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    print("[5.8]  MACD, MACD_Signal, MACD_Hist — trend/momentum signals")

    # ── 5.9 Average True Range (ATR — volatility for risk sizing) ──────────
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()
    print("[5.9]  ATR_14            — Average True Range (stop-loss sizing)")

    # ── 5.10 Price-to-SMA ratio (mean-reversion signal) ───────────────────
    df["Price_to_SMA20"] = df["Close"] / df["SMA_20"]
    print("[5.10] Price_to_SMA20   — mean-reversion z-score proxy")

    # ── 5.11 Vectorised Trend Signal via np.where ──────────────────────────
    #  Rule: Close > SMA_20  AND  SMA_20 > SMA_50  → confirmed uptrend
    #  np.where is C-level vectorisation — no Python loop overhead
    df["Trend"] = np.where(
        (df["Close"] > df["SMA_20"]) & (df["SMA_20"] > df["SMA_50"]),
        "Up",
        "Down",
    )
    le = LabelEncoder()
    df["Trend_Label"] = le.fit_transform(df["Trend"])   # Up=1, Down=0
    print("[5.11] Trend / Trend_Label — np.where vectorised signal")

    # ── 5.12 Volume z-score (anomalous volume = informed trading) ──────────
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std  = df["Volume"].rolling(20).std()
    df["Volume_ZScore"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    print("[5.12] Volume_ZScore     — abnormal volume detection")

    print(f"\n[5.13] Shape after feature engineering : {df.shape}")

    # ── 5.13 Visualise engineered features ────────────────────────────────
    _plot_engineered_features(df)

    return df


def _plot_engineered_features(df: pd.DataFrame) -> None:
    """Publication-quality multi-panel time-series chart."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    fig.suptitle("Engineered Financial Features — Time-Series View",
                 fontsize=16, color="#e0e0e0", fontweight="bold", y=0.98)

    # Panel 1 — Price & Moving Averages with Bollinger Bands
    ax = axes[0]
    ax.plot(df.index, df["Close"],   color="#42a5f5", lw=1.2, label="Close",   alpha=0.9)
    ax.plot(df.index, df["SMA_20"],  color="#ffca28", lw=1.5, label="SMA 20",  ls="--")
    ax.plot(df.index, df["SMA_50"],  color="#ef5350", lw=1.5, label="SMA 50",  ls=":")
    ax.fill_between(df.index, df["BB_Upper"], df["BB_Lower"],
                    alpha=0.12, color="#7c4dff", label="Bollinger Bands")
    ax.set_ylabel("Price (USD)", color="#e0e0e0")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax.set_title("Price + Moving Averages + Bollinger Bands", color="#e0e0e0", pad=6)

    # Panel 2 — Daily Return with colour coding
    ax = axes[1]
    colors = ["#66bb6a" if r >= 0 else "#ef5350" for r in df["Daily_Return_Pct"].fillna(0)]
    ax.bar(df.index, df["Daily_Return_Pct"].fillna(0), color=colors,
           width=1, alpha=0.8)
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    ax.set_ylabel("Daily Return (%)", color="#e0e0e0")
    ax.set_title("Daily Return % (green=gain, red=loss)", color="#e0e0e0", pad=6)

    # Panel 3 — RSI with overbought/oversold bands
    ax = axes[2]
    ax.plot(df.index, df["RSI_14"], color="#ab47bc", lw=1.2, label="RSI 14")
    ax.axhline(70, color="#ef5350", lw=1, ls="--", label="Overbought (70)")
    ax.axhline(30, color="#66bb6a", lw=1, ls="--", label="Oversold (30)")
    ax.fill_between(df.index, 70, 100, alpha=0.08, color="#ef5350")
    ax.fill_between(df.index, 0,   30, alpha=0.08, color="#66bb6a")
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI", color="#e0e0e0")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax.set_title("RSI-14 — Momentum Oscillator", color="#e0e0e0", pad=6)

    # Panel 4 — MACD histogram
    ax = axes[3]
    macd_colors = ["#66bb6a" if v >= 0 else "#ef5350"
                   for v in df["MACD_Hist"].fillna(0)]
    ax.bar(df.index, df["MACD_Hist"].fillna(0), color=macd_colors, width=1, alpha=0.75)
    ax.plot(df.index, df["MACD"],        color="#42a5f5", lw=1.1, label="MACD")
    ax.plot(df.index, df["MACD_Signal"], color="#ffca28", lw=1.1, label="Signal")
    ax.axhline(0, color="#aaa", lw=0.7, ls="--")
    ax.set_ylabel("MACD", color="#e0e0e0")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax.set_title("MACD — Trend + Momentum Convergence", color="#e0e0e0", pad=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/step5_engineered_features.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — FEATURE SELECTION
#  WHY: Multicollinearity inflates model variance and degrades
#       interpretability. SMA_10/SMA_20/SMA_50 and Close are near-perfectly
#       correlated (r ≈ 0.99). Keeping all of them is redundant and will
#       cause coefficient instability in any regularised model.
#  Strategy:
#    a) Drop low-variance features (VarianceThreshold)
#    b) Drop high-correlation pairs (Pearson |r| > 0.90)
#    c) Drop raw OHLCV after deriving features from them (no information gain)
#    d) Drop NaN-heavy rows introduced by rolling windows
# ══════════════════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame, corr_threshold: float = 0.90) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  STEP 6 — FEATURE SELECTION")
    print("═" * 60)

    df = df.copy()

    # ── 6.1 Drop rows with NaN from rolling windows (first ~50 rows) ───────
    n_before = len(df)
    df.dropna(inplace=True)
    print(f"[6.1]  Rows dropped (rolling NaN warmup) : {n_before - len(df)}")
    print(f"       Remaining rows                    : {len(df)}")

    # ── 6.2 Keep only numeric + the target label ───────────────────────────
    keep_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[keep_cols].copy()

    # ── 6.3 Remove low-variance features (near-constant columns) ──────────
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df_numeric)
    low_var_mask   = ~selector.get_support()
    low_var_cols   = df_numeric.columns[low_var_mask].tolist()
    df_numeric     = df_numeric.loc[:, selector.get_support()]
    print(f"\n[6.2]  Low-variance columns dropped : {low_var_cols if low_var_cols else 'None'}")

    # ── 6.4 Remove high-correlation pairs ─────────────────────────────────
    corr_matrix  = df_numeric.corr().abs()
    upper_tri    = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols    = [col for col in upper_tri.columns
                    if any(upper_tri[col] > corr_threshold)]
    df_numeric.drop(columns=drop_cols, inplace=True)
    print(f"[6.3]  High-corr columns dropped (|r|>{corr_threshold}) :")
    for c in drop_cols:
        print(f"       • {c}")

    # ── 6.5 Visualise final correlation matrix ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11))
    corr_final = df_numeric.corr()
    mask = np.triu(np.ones_like(corr_final, dtype=bool))
    sns.heatmap(
        corr_final, mask=mask, ax=ax, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#0f0f0f",
        annot_kws={"size": 8},
    )
    ax.set_title("Final Feature Correlation Matrix — Model-Ready Dataset",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    ax.tick_params(colors="#e0e0e0", labelsize=8)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/step6_final_correlation.png"
    plt.savefig(path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    print(f"[SAVED] {path}")

    print(f"\n[6.4]  ✓ FINAL MODEL-READY DATASET")
    print(f"       Shape  : {df_numeric.shape}")
    print(f"       Columns: {list(df_numeric.columns)}")

    return df_numeric


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    print("\n" + "█" * 60)
    print("  Financial Dataset Analyzer & Feature Engineer")
    print("  Production-Grade Data Science Pipeline")
    print("█" * 60)

    # Step 0 — Generate (or load) dataset
    raw_df = generate_synthetic_dataset()

    # Step 1 — Load & isolate
    df_raw = load_and_isolate(raw_df, ticker="AAPL")

    # Step 2 — EDA
    run_eda(df_raw)

    # Step 3 — Clean
    df_clean = clean_data(df_raw)

    # Step 4 — Preprocess
    df_preprocessed, scaler = preprocess_data(df_clean)

    # Step 5 — Feature engineering
    df_featured = engineer_features(df_preprocessed)

    # Step 6 — Feature selection → model-ready
    df_final = select_features(df_featured)

    # ── Save final artefact ────────────────────────────────────────────────
    output_path = f"{OUTPUT_DIR}/model_ready_dataset.csv"
    df_final.to_csv(output_path)
    print(f"\n[✓] Model-ready dataset saved → {output_path}")
    print(f"    Shape : {df_final.shape}")
    print(f"\n    Head  :\n{df_final.head(3).to_string()}")

    return df_final


if __name__ == "__main__":
    df_model_ready = main()
