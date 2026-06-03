import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# ─── GLOBAL MATPLOTLIB THEME ─────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#161b27",
    "axes.edgecolor":   "#2a3247",
    "axes.labelcolor":  "#a0aec0",
    "xtick.color":      "#a0aec0",
    "ytick.color":      "#a0aec0",
    "text.color":       "#e2e8f0",
    "grid.color":       "#2a3247",
    "grid.linewidth":   0.6,
    "axes.titlecolor":  "#e2e8f0",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "font.family":      "monospace",
})

ACCENT   = "#3b82f6"   # blue-500
ACCENT2  = "#06b6d4"   # cyan-500
SUCCESS  = "#10b981"   # emerald-500
WARN     = "#f59e0b"   # amber-500
DANGER   = "#ef4444"   # red-500
MUTED    = "#64748b"

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DBD Predictor",
    page_icon="assets/icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #0f1525 !important;
    border-right: 1px solid #1e2a40;
}
[data-testid="block-container"] {
    padding: 2rem 3rem 4rem 3rem;
}

/* ── Sidebar Logo Area ── */
.sidebar-brand {
    padding: 1.5rem 0 0.5rem 0;
    text-align: center;
}
.sidebar-brand .brand-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #1d4ed8, #0891b2);
    border-radius: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.75rem;
}
.sidebar-brand .brand-name {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #f1f5f9;
}
.sidebar-brand .brand-sub {
    font-size: 0.7rem;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1e2a40;
    margin: 1rem 0;
}

/* ── Navigation Radio ── */
[data-testid="stRadio"] > div {
    gap: 0.25rem !important;
}
[data-testid="stRadio"] label {
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    transition: background 0.15s;
    color: #94a3b8 !important;
    font-size: 0.85rem;
}
[data-testid="stRadio"] label:hover {
    background: #1a2236;
    color: #e2e8f0 !important;
}

/* ── Page Header ── */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1e2a40;
}
.page-header .page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0;
}
.page-header .page-subtitle {
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 0.25rem;
    font-family: 'DM Mono', monospace;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #111827;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
}
.metric-card .mc-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.5rem;
}
.metric-card .mc-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.metric-card .mc-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
}

/* ── Section Headers ── */
.section-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #cbd5e1;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2a40;
}

/* ── Info / Warning Banners ── */
.info-banner {
    background: #0c1a2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 0.875rem 1.25rem;
    font-size: 0.85rem;
    color: #93c5fd;
    margin-bottom: 1rem;
    font-family: 'DM Mono', monospace;
}
.warn-banner {
    background: #1a1200;
    border: 1px solid #3d2a00;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.875rem 1.25rem;
    font-size: 0.85rem;
    color: #fcd34d;
    margin-bottom: 1rem;
    font-family: 'DM Mono', monospace;
}
.success-banner {
    background: #011a10;
    border: 1px solid #064e35;
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 0.875rem 1.25rem;
    font-size: 0.85rem;
    color: #6ee7b7;
    margin-bottom: 1rem;
    font-family: 'DM Mono', monospace;
}

/* ── Form Styling ── */
[data-testid="stForm"] {
    background: #111827;
    border: 1px solid #1e2a40;
    border-radius: 16px;
    padding: 1.5rem !important;
}
input[type="number"], select {
    background: #0d1324 !important;
    border: 1px solid #1e2a40 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Primary Button ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1d4ed8, #0891b2) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 1.5rem !important;
    color: #fff !important;
    transition: opacity 0.2s, transform 0.1s;
}
[data-testid="baseButton-primary"]:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e2a40;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1.5px dashed #1e3a5f !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

/* ── Prediction result card ── */
.pred-card {
    background: linear-gradient(135deg, #0c1a2e 0%, #0d1f38 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1d4ed8, #0891b2, #10b981);
}
.pred-card .pred-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #64748b;
    font-family: 'DM Mono', monospace;
}
.pred-card .pred-value {
    font-size: 4rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.pred-card .pred-unit {
    font-size: 0.9rem;
    color: #94a3b8;
}

/* ── Model Architecture ── */
.arch-card {
    background: #111827;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
}
.arch-card .arch-role {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #64748b;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.5rem;
}
.arch-card .arch-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
}
.arch-card .arch-params {
    font-size: 0.75rem;
    color: #475569;
    font-family: 'DM Mono', monospace;
    margin-top: 0.25rem;
}

/* ── Stat table ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1e2a40;
    font-size: 0.85rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-row .stat-key { color: #64748b; font-family: 'DM Mono', monospace; }
.stat-row .stat-val { color: #e2e8f0; font-weight: 600; font-family: 'DM Mono', monospace; }

/* ── Streamlit default overrides ── */
.stMetric { background: #111827; border-radius: 10px; padding: 1rem; border: 1px solid #1e2a40; }
div[data-testid="metric-container"] { background: #111827 !important; }
.stAlert { border-radius: 10px; }
footer { display: none; }
#MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────────
for k in ["df", "df_scaled", "model", "scaler", "le_kelamin", "le_demam",
          "X_test", "y_test", "y_pred", "features", "trained"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.trained is None:
    st.session_state.trained = False

# ─── HELPERS ────────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">{title}</div>
        {'<div class="page-subtitle">' + subtitle + '</div>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)

def metric_cards(items):
    """items: list of (label, value, sub)"""
    cards = ""
    for label, value, sub in items:
        cards += f"""
        <div class="metric-card">
            <div class="mc-label">{label}</div>
            <div class="mc-value">{value}</div>
            {'<div class="mc-sub">' + sub + '</div>' if sub else ''}
        </div>"""
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def info(msg):
    st.markdown(f'<div class="info-banner">{msg}</div>', unsafe_allow_html=True)

def success(msg):
    st.markdown(f'<div class="success-banner">{msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-banner">{msg}</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes, filename):
    try:
        if "xls" in filename.lower() and not filename.endswith(".xlsx"):
            df = pd.read_csv(file_bytes, sep="\t")
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_excel(file_bytes)
    except Exception:
        df = pd.read_csv(file_bytes, sep=",")

    junk = ["Unnamed: 9","Unnamed: 10","preprocessing",
            "Normalisasi, cleaning, missing value, duplikate data, balancing, transformasi",
            "Unnamed: 13"]
    df.drop(columns=[c for c in junk if c in df.columns], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    le_k = LabelEncoder(); le_d = LabelEncoder()
    df["jenis_kelamin_enc"] = le_k.fit_transform(df["jenis_kelamin"])
    df["jenis_demam_enc"]   = le_d.fit_transform(df["jenis_demam"])

    df["tgl_masuk"]  = pd.to_datetime(df["tgl_masuk"],  dayfirst=True)
    df["tgl_keluar"] = pd.to_datetime(df["tgl_keluar"], dayfirst=True)
    mn = df["tgl_masuk"].min()
    df["tgl_masuk_num"]  = (df["tgl_masuk"]  - mn).dt.days
    df["tgl_keluar_num"] = (df["tgl_keluar"] - mn).dt.days

    scaler = MinMaxScaler()
    num_cols = ["umur","hemoglobin","hct","trombosit","tgl_masuk_num","tgl_keluar_num"]
    df_s = df.copy()
    df_s[num_cols] = scaler.fit_transform(df[num_cols])
    return df, df_s, scaler, le_k, le_d

def train_model(df_scaled):
    features = ["umur","hemoglobin","hct","trombosit","jenis_kelamin_enc","jenis_demam_enc"]
    X = df_scaled[features]; y = df_scaled["lama_dirawat"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    base = [
        ("lin_reg", LinearRegression()),
        ("knn_reg", KNeighborsRegressor(n_neighbors=5)),
        ("svm_reg", SVR(kernel="rbf")),
        ("ann_reg", MLPRegressor(hidden_layer_sizes=(100,), max_iter=4000, random_state=42)),
    ]
    model = StackingRegressor(estimators=base, final_estimator=LinearRegression(), cv=5)
    model.fit(Xtr, ytr)
    return model, features, Xte, yte, model.predict(Xte)

def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
        </div>
        <div class="brand-name">DBD Predictor</div>
        <div class="brand-sub">Kelompok 7 &mdash; KB A</div>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["Upload & Preprocessing", "Exploratory Data Analysis",
         "Training Model", "Prediksi Pasien", "Evaluasi Model"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0.75rem; background:#111827; border-radius:10px; border:1px solid #1e2a40;">
        <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.12em; color:#475569; font-family:'DM Mono',monospace; margin-bottom:0.5rem;">Dataset Info</div>
        <div style="font-size:0.78rem; color:#64748b; font-family:'DM Mono',monospace; line-height:1.7;">
            Target &nbsp;&nbsp; lama_dirawat<br>
            Task &nbsp;&nbsp;&nbsp;&nbsp; Regression<br>
            Model &nbsp;&nbsp;&nbsp; Stacking
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
if nav == "Upload & Preprocessing":
    page_header("Upload & Preprocessing", "Muat dataset dan terapkan pipeline preprocessing otomatis")

    uploaded = st.file_uploader(
        "Dataset", type=["csv","xls","xlsx"],
        label_visibility="collapsed",
        help="Format: .csv atau .xls (tab-separated)"
    )

    if uploaded:
        with st.spinner("Memproses..."):
            df, df_s, scaler, le_k, le_d = load_and_preprocess(uploaded, uploaded.name)
            st.session_state.update({"df":df,"df_scaled":df_s,"scaler":scaler,
                                     "le_kelamin":le_k,"le_demam":le_d,"trained":False})

        success(f"Dataset berhasil dimuat &mdash; {df.shape[0]} baris, {df.shape[1]} kolom")

        metric_cards([
            ("Total Baris",    str(df.shape[0]), None),
            ("Total Kolom",    str(df.shape[1]), None),
            ("Missing Value",  str(df.isnull().sum().sum()), "setelah cleaning"),
            ("Duplikat",       "0", "setelah cleaning"),
        ])

        section("Preview Data")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

        section("Statistik Deskriptif")
        st.dataframe(df.describe().round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            section("Encoding — Jenis Kelamin")
            for cls, val in zip(le_k.classes_, le_k.transform(le_k.classes_)):
                st.markdown(f'<div class="stat-row"><span class="stat-key">{cls}</span><span class="stat-val">{val}</span></div>', unsafe_allow_html=True)
        with col2:
            section("Encoding — Jenis Demam")
            for cls, val in zip(le_d.classes_, le_d.transform(le_d.classes_)):
                st.markdown(f'<div class="stat-row"><span class="stat-key">{cls}</span><span class="stat-val">{val}</span></div>', unsafe_allow_html=True)

        section("Hasil Normalisasi — 5 Baris Pertama")
        num_cols = ["umur","hemoglobin","hct","trombosit","tgl_masuk_num","tgl_keluar_num"]
        st.dataframe(df_s[num_cols].head().round(6), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; border:1.5px dashed #1e3a5f; border-radius:16px; background:#0d1324;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1e3a5f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:1rem;">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div style="color:#334155; font-size:0.85rem; font-family:'DM Mono',monospace;">Unggah file dataset untuk memulai</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "Exploratory Data Analysis":
    page_header("Exploratory Data Analysis", "Visualisasi distribusi, korelasi, dan pola tersembunyi pada data")

    if st.session_state.df is None:
        warn("Upload dataset terlebih dahulu melalui menu Upload & Preprocessing.")
    else:
        df = st.session_state.df
        df_s = st.session_state.df_scaled
        feat_num = ["umur","hemoglobin","hct","trombosit"]
        target   = "lama_dirawat"

        # Distribusi klinis
        section("Distribusi Data Klinis")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax in axes: ax.grid(True, alpha=0.4)
        sns.histplot(df["umur"],      kde=True, color=ACCENT,  ax=axes[0], edgecolor="none")
        axes[0].set_title("Distribusi Umur")
        sns.histplot(df["trombosit"], kde=True, color=ACCENT2, ax=axes[1], edgecolor="none")
        axes[1].set_title("Distribusi Trombosit")
        counts = df["jenis_demam"].value_counts()
        axes[2].bar(counts.index, counts.values, color=[ACCENT, ACCENT2, SUCCESS], width=0.5)
        axes[2].set_title("Frekuensi Jenis Demam")
        fig.tight_layout(pad=2)
        fig_to_st(fig)

        # Distribusi target
        section("Distribusi Target — Lama Dirawat")
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax in axes2: ax.grid(True, alpha=0.4)
        sns.boxplot(x=df[target], color=SUCCESS, ax=axes2[0], linecolor="#e2e8f0", linewidth=1)
        axes2[0].set_title("Boxplot Lama Dirawat")
        sns.violinplot(x=df[target], color=SUCCESS, ax=axes2[1], linecolor="#e2e8f0", linewidth=1)
        axes2[1].set_title("Violin Plot Lama Dirawat")
        fig2.tight_layout(pad=2)
        fig_to_st(fig2)

        # Rata-rata per jenis demam
        section("Rata-rata Lama Dirawat per Jenis Demam")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.grid(True, alpha=0.4, axis="y")
        palette = {k: c for k, c in zip(df["jenis_demam"].unique(), [ACCENT, ACCENT2, SUCCESS, WARN])}
        sns.barplot(data=df, x="jenis_demam", y=target, estimator=np.mean,
                    hue="jenis_demam", palette=palette, legend=False, ax=ax3)
        ax3.set_title("Rata-rata Lama Dirawat berdasarkan Jenis Demam")
        ax3.set_ylabel("Rata-rata Hari"); ax3.set_xlabel("")
        fig3.tight_layout()
        fig_to_st(fig3)

        # Korelasi & Heatmap
        col_l, col_r = st.columns([3, 2])
        with col_l:
            section("Heatmap Korelasi Pearson")
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(df_s[feat_num + [target]].corr(), annot=True, fmt=".2f",
                        cmap=cmap, center=0, ax=ax4,
                        linewidths=0.5, linecolor="#0f1117",
                        annot_kws={"size": 10, "family": "monospace"})
            ax4.set_title("Korelasi Pearson")
            fig4.tight_layout()
            fig_to_st(fig4)

        with col_r:
            section("Tabel Korelasi")
            pearson  = df_s[feat_num + [target]].corr(method="pearson")[target].drop(target)
            spearman = df_s[feat_num + [target]].corr(method="spearman")[target].drop(target)
            r_sc = {}
            for col in feat_num:
                m = LinearRegression()
                m.fit(df_s[[col]], df_s[target])
                r_sc[col] = m.score(df_s[[col]], df_s[target])
            results = pd.DataFrame({"Pearson": pearson, "Spearman": spearman,
                                    "R²": pd.Series(r_sc)})
            st.dataframe(results.round(4), use_container_width=True)

            section("Statistik Lanjutan")
            cont = pd.crosstab(df["jenis_demam"], df[target])
            chi2, p, _, _ = chi2_contingency(cont)
            X_cl = df_s[feat_num]
            km   = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_cl)
            sil  = silhouette_score(X_cl, km.labels_)
            for label, val in [("Chi² Statistic", f"{chi2:.4f}"), ("p-value", f"{p:.4f}"),
                                ("Silhouette Coeff.", f"{sil:.4f}")]:
                st.markdown(f'<div class="stat-row"><span class="stat-key">{label}</span><span class="stat-val">{val}</span></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "Training Model":
    page_header("Training Model", "Stacking Regressor dengan 4 base learner dan 1 meta learner")

    if st.session_state.df_scaled is None:
        warn("Upload dataset terlebih dahulu.")
    else:
        col_info, col_train = st.columns([2, 1])

        with col_info:
            section("Arsitektur Model")
            arch_items = [
                ("Base Learner 1", "Linear Regression",   "Baseline linier"),
                ("Base Learner 2", "KNN Regressor",       "n_neighbors = 5"),
                ("Base Learner 3", "SVR",                  "kernel = RBF"),
                ("Base Learner 4", "MLP Regressor",       "hidden_layers = (100,) &nbsp;|&nbsp; max_iter = 4000"),
                ("Meta Learner",   "Linear Regression",   "final_estimator, cv = 5-fold"),
            ]
            for role, name, params in arch_items:
                st.markdown(f"""
                <div class="arch-card">
                    <div class="arch-role">{role}</div>
                    <div class="arch-name">{name}</div>
                    <div class="arch-params">{params}</div>
                </div>""", unsafe_allow_html=True)

        with col_train:
            section("Konfigurasi Split")
            n = len(st.session_state.df_scaled)
            metric_cards([
                ("Total Data",    str(n),          None),
                ("Training Set",  str(int(n*0.8)), "80 %"),
                ("Test Set",      str(n - int(n*0.8)), "20 %"),
            ])

            section("Aksi")
            btn = st.button("Mulai Training", type="primary", use_container_width=True)
            if st.session_state.trained:
                success("Model sudah dilatih dan siap digunakan.")

        if btn:
            with st.spinner("Melatih model Stacking Regressor... Mohon tunggu."):
                model, features, Xte, yte, ypred = train_model(st.session_state.df_scaled)
                st.session_state.update({"model": model, "features": features,
                                         "X_test": Xte, "y_test": yte,
                                         "y_pred": ypred, "trained": True})
            success("Training selesai. Navigasikan ke halaman Evaluasi Model untuk melihat hasil.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "Prediksi Pasien":
    page_header("Prediksi Pasien", "Estimasi lama rawat inap berdasarkan parameter klinis pasien")

    if not st.session_state.trained:
        warn("Latih model terlebih dahulu melalui menu Training Model.")
    else:
        le_k    = st.session_state.le_kelamin
        le_d    = st.session_state.le_demam
        scaler  = st.session_state.scaler
        model   = st.session_state.model

        col_form, col_result = st.columns([3, 2])

        with col_form:
            section("Data Pasien")
            with st.form("form_prediksi"):
                c1, c2 = st.columns(2)
                with c1:
                    jenis_kelamin = st.selectbox("Jenis Kelamin", list(le_k.classes_))
                    umur          = st.number_input("Umur (tahun)", 0, 120, 25)
                    jenis_demam   = st.selectbox("Jenis Demam",   list(le_d.classes_))
                with c2:
                    hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 13.0, 0.1)
                    hct        = st.number_input("Hematokrit / HCT (%)", 0.0, 70.0, 40.0, 0.1)
                    trombosit  = st.number_input("Trombosit (×10³/µL)", 0, 1000, 150)
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Hitung Prediksi", type="primary", use_container_width=True)

        with col_result:
            section("Hasil Estimasi")
            if submitted:
                kel_enc = le_k.transform([jenis_kelamin])[0]
                dem_enc = le_d.transform([jenis_demam])[0]
                raw     = np.array([[umur, hemoglobin, hct, trombosit, 0, 0]], dtype=float)
                scaled  = scaler.transform(raw)
                inp     = np.array([[scaled[0][0], scaled[0][1], scaled[0][2], scaled[0][3], kel_enc, dem_enc]])
                pred    = model.predict(inp)[0]
                days    = max(1, round(pred))

                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-label">Estimasi Lama Rawat</div>
                    <div class="pred-value">{days}</div>
                    <div class="pred-unit">hari</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for k, v in [("Jenis Kelamin", jenis_kelamin), ("Umur", f"{umur} tahun"),
                              ("Jenis Demam", jenis_demam), ("Hemoglobin", f"{hemoglobin} g/dL"),
                              ("Hematokrit", f"{hct} %"), ("Trombosit", f"{trombosit} ×10³/µL"),
                              ("Nilai Scaled", f"{pred:.5f}")]:
                    st.markdown(f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center; padding:3rem 1rem; color:#334155; font-family:'DM Mono',monospace; font-size:0.8rem;">
                    Isi form dan klik<br>Hitung Prediksi
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — EVALUASI
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "Evaluasi Model":
    page_header("Evaluasi Model", "Performa Stacking Regressor pada test set (80/20 split)")

    if not st.session_state.trained:
        warn("Latih model terlebih dahulu melalui menu Training Model.")
    else:
        yte  = st.session_state.y_test
        yprd = st.session_state.y_pred

        mse  = mean_squared_error(yte, yprd)
        rmse = np.sqrt(mse)
        mad  = mean_absolute_error(yte, yprd)
        mape = mean_absolute_percentage_error(np.where(yte == 0, 1e-10, yte), yprd)

        metric_cards([
            ("MSE",  f"{mse:.4f}",  "Mean Squared Error"),
            ("RMSE", f"{rmse:.4f}", "Root Mean Squared Error"),
            ("MAD",  f"{mad:.4f}",  "Mean Absolute Deviation"),
            ("MAPE", f"{mape:.4f}", "Mean Absolute Percentage Error"),
        ])

        col_l, col_r = st.columns(2)

        with col_l:
            section("Actual vs Prediction")
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.grid(True, alpha=0.3)
            ax1.plot(yte.values, color=ACCENT,  linewidth=1.5, label="Actual",     alpha=0.9)
            ax1.plot(yprd,       color=ACCENT2, linewidth=1.5, label="Prediction", alpha=0.9)
            ax1.set_title("Actual vs Prediction")
            ax1.set_xlabel("Data Index"); ax1.set_ylabel("Nilai")
            ax1.legend(facecolor="#161b27", edgecolor="#2a3247", labelcolor="#e2e8f0")
            fig1.tight_layout()
            fig_to_st(fig1)

        with col_r:
            section("Scatter — Actual vs Predicted")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.grid(True, alpha=0.3)
            ax2.scatter(yte, yprd, alpha=0.5, color=ACCENT, s=30, edgecolors="none")
            mn = min(float(yte.min()), float(yprd.min()))
            mx = max(float(yte.max()), float(yprd.max()))
            ax2.plot([mn, mx], [mn, mx], color=DANGER, linestyle="--", linewidth=1.2, label="Perfect Fit")
            ax2.set_title("Actual vs Predicted"); ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted")
            ax2.legend(facecolor="#161b27", edgecolor="#2a3247", labelcolor="#e2e8f0")
            fig2.tight_layout()
            fig_to_st(fig2)

        section("Distribusi Residual")
        residuals = yte.values - yprd
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        ax3.grid(True, alpha=0.3, axis="y")
        sns.histplot(residuals, kde=True, color="#8b5cf6", bins=30, ax=ax3,
                     edgecolor="none", alpha=0.7)
        ax3.axvline(x=0, color=DANGER, linestyle="--", linewidth=1.2, label="Error = 0")
        ax3.set_title("Distribusi Residual (Actual − Predicted)")
        ax3.set_xlabel("Residual"); ax3.set_ylabel("Frekuensi")
        ax3.legend(facecolor="#161b27", edgecolor="#2a3247", labelcolor="#e2e8f0")
        fig3.tight_layout()
        fig_to_st(fig3)

        section("Bar Chart Metrik Evaluasi")
        fig4, ax4 = plt.subplots(figsize=(8, 3.5))
        ax4.grid(True, alpha=0.3, axis="y")
        colors = [ACCENT, ACCENT2, SUCCESS, WARN]
        bars = ax4.bar(["MSE","RMSE","MAD","MAPE"], [mse, rmse, mad, mape],
                       color=colors, width=0.5, edgecolor="none")
        for bar, val in zip(bars, [mse, rmse, mad, mape]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=9, color="#94a3b8", fontfamily="monospace")
        ax4.set_title("Evaluasi Stacking Regressor")
        fig4.tight_layout()
        fig_to_st(fig4)
