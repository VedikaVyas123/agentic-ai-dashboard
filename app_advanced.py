# app_advanced.py — polished UI + robust plotting + safer LLM calls

import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# ------------- setup -------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_ENABLED = bool(OPENAI_API_KEY)
client = None
if LLM_ENABLED:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        LLM_ENABLED = False

# Aesthetic style (brighter palette for LinkedIn screenshots)
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.facecolor": "#0e1117",
    "figure.facecolor": "#0e1117",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "axes.edgecolor": "#4c4c4c",
    "grid.color": "#2a2a2a",
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

LILAC = "#a78bfa"      # primary accent
MINT  = "#9ece6a"      # secondary

# ------------- helpers -------------
def read_any_table(f):
    """Robust file reader for Gradio File (csv/xlsx) or path-like."""
    if f is None:
        raise ValueError("Please upload a CSV or Excel file.")
    # If f has raw bytes
    if hasattr(f, "read"):
        data = f.read()
        try:
            # try CSV
            text = data.decode("utf-8")
            return pd.read_csv(io.StringIO(text))
        except Exception:
            # try Excel
            return pd.read_excel(io.BytesIO(data))
    # If Gradio provides a tempfile path
    path = getattr(f, "name", str(f))
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def suggest_time_col(df: pd.DataFrame):
    """Find a likely datetime column if present."""
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in ["date", "time", "timestamp"]):
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
    # fallback: already-parsed datetimes
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def summary_text(df: pd.DataFrame) -> str:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_str = "none" if miss.empty else "\n".join([f"- {c}: {int(v)}" for c, v in miss.items()])
    dtypes_str = ", ".join([f"{k}={v}" for k, v in df.dtypes.astype(str).value_counts().items()])
    return f"**Shape:** {df.shape[0]} × {df.shape[1]}\n\n**Missing values:**\n{miss_str}\n\n**Column types:** {dtypes_str}"

# ------------- plots -------------
def plot_hist(df, target):
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]): 
        return None
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[target].dropna(), kde=True, color=LILAC, ax=ax)
    ax.set_title(f"Distribution of {target}", color="white")
    ax.set_xlabel(target); ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def plot_time(df, target):
    tcol = suggest_time_col(df)
    if not tcol or target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return None
    ts = df[[tcol, target]].dropna().copy()
    # coerce to datetime
    ts[tcol] = pd.to_datetime(ts[tcol], errors="coerce")
    ts = ts.dropna(subset=[tcol]).sort_values(tcol)
    if ts.empty: 
        return None
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(x=ts[tcol], y=ts[target], ax=ax, color=LILAC)
    ax.set_title(f"{target} over time ({tcol})", color="white")
    ax.set_xlabel(tcol); ax.set_ylabel(target)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment('right')
    fig.tight_layout()
    return fig

def plot_corr(df, target):
    if target not in df.columns: 
        return None
    num = df.select_dtypes(include=np.number)
    if target not in num.columns: 
        return None
    corr = num.corr(numeric_only=True)[target].drop(labels=[target]).dropna()
    if corr.empty: 
        return None
    top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:6]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=top.values, y=top.index, ax=ax, color=LILAC)
    ax.set_title(f"Top correlations with {target}", color="white")
    ax.set_xlabel("Pearson r")
    fig.tight_layout()
    return fig

def plot_cat_avg(df, target):
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]): 
        return None
    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]
    if not cat_cols: 
        return None
    c = cat_cols[0]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=df[c], y=df[target], estimator=np.mean, ax=ax, color=MINT)
    ax.set_title(f"Average {target} by {c}", color="white")
    ax.set_xlabel(c); ax.set_ylabel(target)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment('right')
    fig.tight_layout()
    return fig

from ydata_profiling import ProfileReport
def generate_eda_report(df):
    """Generate and save an EDA report."""
    try:
        profile = ProfileReport(df, title="EDA Report - Agentic AI Dashboard", minimal=True)
        profile.to_file("eda_report.html")
        return "✅ EDA report successfully generated: eda_report.html"
    except Exception as e:
        return f"⚠️ Could not generate EDA report: {e}"


# ------------- ML + reflection -------------
def train_baseline(df, target):
    if target not in df.columns:
        return "Select a target column.", None, None

    # Keep numeric features only
    y = df[target]
    X = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")

    # Drop rows with missing target
    mask = ~y.isna()
    X, y = X[mask], y[mask]

    if X.empty or y.empty:
        return "No rows to train on after cleaning.", None, None

    # 🔧 Fix: handle NaN values using mean imputation
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    preds_df = pd.DataFrame({"y_true": y_test.values[:10], "y_pred": y_pred[:10]})
    metrics_md = (
        f"**Baseline Regression (LinearRegression)**\n"
        f"- **MAE:** {mae:.3f}\n"
        f"- **R²:** {r2:.3f}\n\n"
        f"**Sample predictions (first 10):**\n\n{preds_df.to_markdown(index=False)}"
    )

    return metrics_md, mae, r2

def reflect_with_llm(mae, r2):
    if not LLM_ENABLED or client is None:
        return "*LLM reflection disabled — add a valid OPENAI_API_KEY in .env*"
    prompt = f"""You are a concise analytics copilot.
Metrics:
- MAE: {mae:.2f}
- R2: {r2:.3f}

Write ≤90 words:
- 2 crisp insights about model performance in plain English
- 2 next analysis steps
No jargon, bullets preferred."""
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM reflection skipped (API issue): {e}"

# ------------- Gradio UI -------------
with gr.Blocks(
    theme=gr.themes.Default(primary_hue="violet", secondary_hue="pink"),
    css="""
        #root {background-color:#0e1117;}
        .gr-button {border-radius:12px !important;}
        .gr-textbox, .gr-dropdown {background-color:#1e1e2f !important; border-radius:10px;}
        .gr-plot {background-color:#121212 !important; border-radius:8px;}
        h1, h2, h3 {color:#c9d1d9 !important; text-align:center;}
    """
) as demo:
    gr.Markdown(
        "<h1>Agentic AI Dashboard — Analyze • Predict • Reflect</h1>"
        "<p style='text-align:center;color:#94a3b8'>Upload a dataset, choose a target, view polished visuals, train a baseline model, and get an AI reflection.</p>"
    )

    with gr.Row():
        file_in = gr.File(label="Upload CSV / Excel", file_types=[".csv", ".xls", ".xlsx"])
        target_dd = gr.Dropdown(choices=[], label="Target column", interactive=True)

    df_state = gr.State(None)
    summary_md = gr.Markdown()

    with gr.Row():
        analyze_btn = gr.Button("Analyze Target & Plots", variant="primary")
        model_btn   = gr.Button("Train Baseline Model", variant="secondary")

    # Plots
    hist_plot = gr.Plot(label="Target distribution")
    time_plot = gr.Plot(label="Target over time")
    corr_plot = gr.Plot(label="Top correlations")
    cat_plot  = gr.Plot(label="Category averages")

    metrics_md = gr.Markdown()
    reflection_md = gr.Markdown()

    # Footer
    gr.Markdown("<div style='text-align:center;color:#9ba3b0;margin-top:8px'>Built with 💜 by <b>Vedika Vyas</b></div>")

    # ---------- callbacks ----------
    def on_upload(f):
        try:
            df = read_any_table(f)
            # try to parse possible date columns
            for c in df.columns:
                if any(k in c.lower() for k in ["date","time","timestamp"]):
                    df[c] = pd.to_datetime(df[c], errors="ignore")
            choices = list(df.columns)
            default = "revenue" if "revenue" in df.columns else (choices[0] if choices else None)
            return df, gr.update(choices=choices, value=default), summary_text(df)
        except Exception as e:
            return None, gr.update(choices=[], value=None), f"❌ {e}"

    file_in.upload(fn=on_upload, inputs=file_in, outputs=[df_state, target_dd, summary_md])

    def do_plots(df, target):
        if df is None or not target:
            return None, None, None, None
        return plot_hist(df, target), plot_time(df, target), plot_corr(df, target), plot_cat_avg(df, target)

    analyze_btn.click(fn=do_plots, inputs=[df_state, target_dd],
                      outputs=[hist_plot, time_plot, corr_plot, cat_plot])

    def do_model(df, target):
        if df is None or not target:
            return "Upload a file and choose a target.", " "
        metrics, mae, r2 = train_baseline(df, target)
        if mae is None:
            return metrics, " "
        eda_status = generate_eda_report(df)
        print(eda_status) 
        return metrics, reflect_with_llm(mae, r2)

    model_btn.click(fn=do_model, inputs=[df_state, target_dd],
                    outputs=[metrics_md, reflection_md])

if __name__ == "__main__":
    demo.launch(share=True)
