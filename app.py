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

# ─── GLOBAL MATPLOTLIB THEME (light) ────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8fafc",
    "axes.edgecolor":    "#e2e8f0",
    "axes.labelcolor":   "#475569",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#1e293b",
    "grid.color":        "#e2e8f0",
    "grid.linewidth":    0.7,
    "axes.titlecolor":   "#0f172a",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ACCENT  = "#2563eb"   # blue-600
ACCENT2 = "#0891b2"   # cyan-600
SUCCESS = "#059669"   # emerald-600
WARN    = "#d97706"   # amber-600
DANGER  = "#dc2626"   # red-600
PURPLE  = "#7c3aed"   # violet-600

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DBD Predictor",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f1f5f9 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #1e293b;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="block-container"] {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    padding: 1.75rem 0 0.75rem 0;
    text-align: center;
}
.brand-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
    border-radius: 16px;
    display: inline-flex; align-items: center; justify-content: center;
    margin-bottom: 0.875rem;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25);
}
.brand-name {
    font-size: 0.95rem; font-weight: 800;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: #0f172a;
}
.brand-sub {
    font-size: 0.68rem; color: #94a3b8;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}
.sidebar-divider {
    border: none; border-top: 1px solid #f1f5f9; margin: 1rem 0;
}

/* ── Nav Radio ── */
[data-testid="stRadio"] > div { gap: 0.2rem !important; }
[data-testid="stRadio"] label {
    border-radius: 8px; padding: 0.5rem 0.875rem;
    transition: all 0.15s; color: #64748b !important;
    font-size: 0.84rem; font-weight: 500;
}
[data-testid="stRadio"] label:hover {
    background: #eff6ff; color: #2563eb !important;
}

/* ── Sidebar Info Panel ── */
.sidebar-info {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.875rem 1rem;
}
.sidebar-info .si-title {
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.6rem;
}
.sidebar-info .si-row {
    display: flex; justify-content: space-between;
    font-size: 0.77rem; color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    padding: 0.15rem 0; line-height: 1.8;
}
.sidebar-info .si-val { color: #1e293b; font-weight: 500; }

/* ── Page Header ── */
.page-header {
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}
.page-title {
    font-size: 1.8rem; font-weight: 800;
    color: #0f172a; letter-spacing: -0.03em; margin: 0;
}
.page-subtitle {
    font-size: 0.82rem; color: #94a3b8;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.75rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.25rem 1.5rem;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s, transform 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.1);
    transform: translateY(-1px);
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #2563eb, #0891b2);
}
.mc-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;
}
.mc-value {
    font-size: 1.65rem; font-weight: 800;
    color: #0f172a; font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.mc-sub {
    font-size: 0.7rem; color: #cbd5e1;
    margin-top: 0.35rem; font-family: 'JetBrains Mono', monospace;
}

/* ── Section Headers ── */
.section-header {
    font-size: 0.72rem; font-weight: 700;
    color: #94a3b8; letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 2rem 0 0.875rem 0;
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
}
.section-header::after {
    content: ''; flex: 1; height: 1px; background: #e2e8f0;
}

/* ── Banners ── */
.info-banner {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb; border-radius: 8px;
    padding: 0.875rem 1.25rem; font-size: 0.83rem;
    color: #1d4ed8; margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.warn-banner {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 3px solid #d97706; border-radius: 8px;
    padding: 0.875rem 1.25rem; font-size: 0.83rem;
    color: #92400e; margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.success-banner {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 3px solid #059669; border-radius: 8px;
    padding: 0.875rem 1.25rem; font-size: 0.83rem;
    color: #065f46; margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Form ── */
[data-testid="stForm"] {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 1.75rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Button ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb, #0891b2) !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.03em !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="baseButton-primary"]:hover {
    opacity: 0.92 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px; overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: #f8fafc !important;
    border: 1.5px dashed #bfdbfe !important;
    border-radius: 12px !important;
}

/* ── Prediction Card ── */
.pred-card {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border: 1px solid #bfdbfe; border-radius: 18px;
    padding: 2.5rem 2rem; text-align: center;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(37,99,235,0.1);
}
.pred-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #2563eb, #0891b2, #059669);
}
.pred-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.16em; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
.pred-value {
    font-size: 4.5rem; font-weight: 800;
    line-height: 1; margin: 0.5rem 0;
    background: linear-gradient(135deg, #2563eb, #0891b2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-family: 'JetBrains Mono', monospace;
}
.pred-unit { font-size: 0.95rem; color: #64748b; font-weight: 600; }

/* ── Arch Card ── */
.arch-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.1rem 1.25rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s;
}
.arch-card:hover { box-shadow: 0 3px 10px rgba(37,99,235,0.1); }
.arch-role {
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 0.3rem;
}
.arch-name { font-size: 0.9rem; font-weight: 700; color: #0f172a; }
.arch-params {
    font-size: 0.74rem; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem;
}

/* ── Stat Row ── */
.stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.83rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
.stat-val { color: #0f172a; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 4rem 2rem;
    border: 1.5px dashed #bfdbfe; border-radius: 16px;
    background: #f8fafc;
}
.empty-state-text {
    color: #cbd5e1; font-size: 0.83rem;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.75rem;
}

/* ── Misc overrides ── */
footer { display: none; }
#MainMenu { display: none; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k in ["df","df_scaled","model","scaler","le_kelamin","le_demam",
          "X_test","y_test","y_pred","features","trained"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.trained is None:
    st.session_state.trained = False

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="page-header"><div class="page-title">{title}</div>{sub_html}</div>',
                unsafe_allow_html=True)

def metric_cards(items):
    cards = ""
    for lbl, val, sub in items:
        sub_html = f'<div class="mc-sub">{sub}</div>' if sub else ""
        cards += (
            '<div class="metric-card">'
            f'<div class="mc-label">{lbl}</div>'
            f'<div class="mc-value">{val}</div>'
            f'{sub_html}'
            '</div>'
        )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def info(msg):
    st.markdown(f'<div class="info-banner">{msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-banner">{msg}</div>', unsafe_allow_html=True)

def success(msg):
    st.markdown(f'<div class="success-banner">{msg}</div>', unsafe_allow_html=True)

def stat_row(key, val):
    st.markdown(f'<div class="stat-row"><span class="stat-key">{key}</span><span class="stat-val">{val}</span></div>',
                unsafe_allow_html=True)

def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
    df.dropna(inplace=True); df.drop_duplicates(inplace=True)

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

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
                 stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
        </div>
        <div class="brand-name">DBD Predictor</div>
        <div class="brand-sub">Kelompok 7 &mdash; KB A</div>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    nav = st.radio(
        "nav", label_visibility="collapsed",
        options=["Upload & Preprocessing","Exploratory Data Analysis",
                 "Training Model","Prediksi Pasien","Evaluasi Model"],
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
        <div class="si-title">Dataset Info</div>
        <div class="si-row"><span>Target</span><span class="si-val">lama_dirawat</span></div>
        <div class="si-row"><span>Task</span><span class="si-val">Regression</span></div>
        <div class="si-row"><span>Model</span><span class="si-val">Stacking</span></div>
        <div class="si-row"><span>CV</span><span class="si-val">5-fold</span></div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "Upload & Preprocessing":
    page_header("Upload & Preprocessing",
                "Muat dataset dan terapkan pipeline preprocessing otomatis")

    uploaded = st.file_uploader("Dataset", type=["csv","xls","xlsx"],
                                label_visibility="collapsed")

    if uploaded:
        with st.spinner("Memproses dataset..."):
            df, df_s, scaler, le_k, le_d = load_and_preprocess(uploaded, uploaded.name)
            st.session_state.update({"df":df,"df_scaled":df_s,"scaler":scaler,
                                     "le_kelamin":le_k,"le_demam":le_d,"trained":False})

        success(f"Dataset berhasil dimuat &mdash; {df.shape[0]} baris, {df.shape[1]} kolom")
        metric_cards([
            ("Total Baris",   str(df.shape[0]), None),
            ("Total Kolom",   str(df.shape[1]), None),
            ("Missing Value", "0", "setelah cleaning"),
            ("Duplikat",      "0", "setelah cleaning"),
        ])

        section("Preview Data")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

        section("Statistik Deskriptif")
        st.dataframe(df.describe().round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            section("Encoding — Jenis Kelamin")
            for cls, val in zip(le_k.classes_, le_k.transform(le_k.classes_)):
                stat_row(cls, str(val))
        with col2:
            section("Encoding — Jenis Demam")
            for cls, val in zip(le_d.classes_, le_d.transform(le_d.classes_)):
                stat_row(cls, str(val))

        section("Hasil Normalisasi — 5 Baris Pertama")
        num_cols = ["umur","hemoglobin","hct","trombosit","tgl_masuk_num","tgl_keluar_num"]
        st.dataframe(df_s[num_cols].head().round(6), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
                 stroke="#bfdbfe" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div class="empty-state-text">Unggah file dataset untuk memulai</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Exploratory Data Analysis":
    page_header("Exploratory Data Analysis",
                "Visualisasi distribusi, korelasi, dan pola tersembunyi pada data")

    if st.session_state.df is None:
        warn("Upload dataset terlebih dahulu melalui menu Upload & Preprocessing.")
    else:
        df   = st.session_state.df
        df_s = st.session_state.df_scaled
        feat_num = ["umur","hemoglobin","hct","trombosit"]
        target   = "lama_dirawat"
        palette  = [ACCENT, ACCENT2, SUCCESS, WARN]

        section("Distribusi Data Klinis")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        sns.histplot(df["umur"],      kde=True, color=ACCENT,  ax=axes[0], edgecolor="none", alpha=0.8)
        axes[0].set_title("Distribusi Umur"); axes[0].set_xlabel("Umur (tahun)")
        sns.histplot(df["trombosit"], kde=True, color=ACCENT2, ax=axes[1], edgecolor="none", alpha=0.8)
        axes[1].set_title("Distribusi Trombosit"); axes[1].set_xlabel("Trombosit")
        counts = df["jenis_demam"].value_counts()
        bars = axes[2].bar(counts.index, counts.values,
                           color=palette[:len(counts)], width=0.5, edgecolor="none")
        axes[2].set_title("Frekuensi Jenis Demam")
        for b in bars:
            axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                         str(int(b.get_height())), ha="center", fontsize=10, color="#475569")
        fig.tight_layout(pad=2.5)
        fig_to_st(fig)

        section("Distribusi Target — Lama Dirawat")
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
        sns.boxplot(x=df[target], color=SUCCESS, ax=axes2[0],
                    linecolor="#065f46", linewidth=1.2, flierprops=dict(marker="o", color=SUCCESS, alpha=0.4))
        axes2[0].set_title("Boxplot Lama Dirawat")
        sns.violinplot(x=df[target], color=SUCCESS, ax=axes2[1],
                       linecolor="#065f46", linewidth=1.2, inner="quartile")
        axes2[1].set_title("Violin Plot Lama Dirawat")
        fig2.tight_layout(pad=2.5)
        fig_to_st(fig2)

        section("Rata-rata Lama Dirawat per Jenis Demam")
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        pal = {k: c for k, c in zip(df["jenis_demam"].unique(), palette)}
        sns.barplot(data=df, x="jenis_demam", y=target, estimator=np.mean,
                    hue="jenis_demam", palette=pal, legend=False, ax=ax3,
                    edgecolor="none", alpha=0.85)
        ax3.set_title("Rata-rata Lama Dirawat berdasarkan Jenis Demam")
        ax3.set_ylabel("Rata-rata Hari"); ax3.set_xlabel("")
        fig3.tight_layout()
        fig_to_st(fig3)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            section("Heatmap Korelasi Pearson")
            fig4, ax4 = plt.subplots(figsize=(7, 5.5))
            cmap = sns.light_palette("#2563eb", as_cmap=True)
            sns.heatmap(df_s[feat_num + [target]].corr(), annot=True, fmt=".2f",
                        cmap=cmap, ax=ax4, linewidths=1, linecolor="#f1f5f9",
                        annot_kws={"size": 10})
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
            for lbl, val in [("Chi² Statistic", f"{chi2:.4f}"),
                              ("p-value",        f"{p:.4f}"),
                              ("Silhouette",     f"{sil:.4f}")]:
                stat_row(lbl, val)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Training Model":
    page_header("Training Model",
                "Stacking Regressor dengan 4 base learner dan 1 meta learner")

    if st.session_state.df_scaled is None:
        warn("Upload dataset terlebih dahulu.")
    else:
        col_info, col_right = st.columns([2, 1])

        with col_info:
            section("Arsitektur Model")
            for role, name, params in [
                ("Base Learner 1", "Linear Regression",  "Baseline linier"),
                ("Base Learner 2", "KNN Regressor",      "n_neighbors = 5"),
                ("Base Learner 3", "SVR",                 "kernel = RBF"),
                ("Base Learner 4", "MLP Regressor",      "hidden_layers = (100,) &nbsp;|&nbsp; max_iter = 4000"),
                ("Meta Learner",   "Linear Regression",  "final_estimator &nbsp;|&nbsp; cv = 5-fold"),
            ]:
                st.markdown(f"""
                <div class="arch-card">
                    <div class="arch-role">{role}</div>
                    <div class="arch-name">{name}</div>
                    <div class="arch-params">{params}</div>
                </div>""", unsafe_allow_html=True)

        with col_right:
            n = len(st.session_state.df_scaled)
            section("Konfigurasi Split")
            metric_cards([
                ("Total Data",   str(n),                 None),
                ("Training Set", str(int(n * 0.8)),      "80 %"),
                ("Test Set",     str(n - int(n * 0.8)),  "20 %"),
            ])
            section("Aksi")
            if st.session_state.trained:
                success("Model sudah dilatih dan siap digunakan.")
            btn = st.button("Mulai Training", type="primary", use_container_width=True)

        if btn:
            with st.spinner("Melatih Stacking Regressor... Mohon tunggu."):
                model, features, Xte, yte, ypred = train_model(st.session_state.df_scaled)
                st.session_state.update({"model":model,"features":features,
                                         "X_test":Xte,"y_test":yte,
                                         "y_pred":ypred,"trained":True})
            success("Training selesai. Buka halaman Evaluasi Model untuk melihat hasil.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Prediksi Pasien":
    page_header("Prediksi Pasien",
                "Estimasi lama rawat inap berdasarkan parameter klinis pasien")

    if not st.session_state.trained:
        warn("Latih model terlebih dahulu melalui menu Training Model.")
    else:
        le_k   = st.session_state.le_kelamin
        le_d   = st.session_state.le_demam
        scaler = st.session_state.scaler
        model  = st.session_state.model

        col_form, col_result = st.columns([3, 2], gap="large")

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
                    trombosit  = st.number_input("Trombosit (x10³/µL)", 0, 1000, 150)
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Hitung Prediksi",
                                                   type="primary", use_container_width=True)

        with col_result:
            section("Hasil Estimasi")
            if submitted:
                kel_enc = le_k.transform([jenis_kelamin])[0]
                dem_enc = le_d.transform([jenis_demam])[0]
                raw     = np.array([[umur, hemoglobin, hct, trombosit, 0, 0]], dtype=float)
                scaled  = scaler.transform(raw)
                inp     = np.array([[scaled[0][0], scaled[0][1], scaled[0][2],
                                     scaled[0][3], kel_enc, dem_enc]])
                pred    = model.predict(inp)[0]
                days    = max(1, round(pred))

                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-label">Estimasi Lama Rawat</div>
                    <div class="pred-value">{days}</div>
                    <div class="pred-unit">hari</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for k, v in [("Jenis Kelamin", jenis_kelamin),
                              ("Umur",          f"{umur} tahun"),
                              ("Jenis Demam",   jenis_demam),
                              ("Hemoglobin",    f"{hemoglobin} g/dL"),
                              ("Hematokrit",    f"{hct} %"),
                              ("Trombosit",     f"{trombosit} x10³/µL"),
                              ("Nilai Scaled",  f"{pred:.5f}")]:
                    stat_row(k, v)
            else:
                st.markdown("""
                <div style="text-align:center;padding:3.5rem 1rem;
                     color:#cbd5e1;font-family:'JetBrains Mono',monospace;font-size:0.78rem;">
                    Isi form dan klik<br>Hitung Prediksi
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EVALUASI
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Evaluasi Model":
    page_header("Evaluasi Model",
                "Performa Stacking Regressor pada test set (80/20 split)")

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
            ("MAPE", f"{mape:.4f}", "Mean Abs. Percentage Error"),
        ])

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            section("Actual vs Prediction")
            fig1, ax1 = plt.subplots(figsize=(7, 4.5))
            ax1.plot(yte.values, color=ACCENT,  linewidth=1.8, label="Actual",     alpha=0.9)
            ax1.plot(yprd,       color=ACCENT2, linewidth=1.8, label="Prediction", alpha=0.9)
            ax1.fill_between(range(len(yte)), yte.values, yprd, alpha=0.07, color=ACCENT)
            ax1.set_title("Actual vs Prediction")
            ax1.set_xlabel("Data Index"); ax1.set_ylabel("Nilai")
            ax1.legend(framealpha=0.9)
            ax1.grid(True, alpha=0.5)
            fig1.tight_layout()
            fig_to_st(fig1)

        with col_r:
            section("Scatter — Actual vs Predicted")
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            ax2.scatter(yte, yprd, alpha=0.55, color=ACCENT, s=35, edgecolors="none")
            mn = min(float(yte.min()), float(yprd.min()))
            mx = max(float(yte.max()), float(yprd.max()))
            ax2.plot([mn, mx], [mn, mx], color=DANGER, linestyle="--",
                     linewidth=1.4, label="Perfect Fit")
            ax2.set_title("Actual vs Predicted (Scatter)")
            ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted")
            ax2.legend(framealpha=0.9)
            ax2.grid(True, alpha=0.5)
            fig2.tight_layout()
            fig_to_st(fig2)

        section("Distribusi Residual")
        residuals = yte.values - yprd
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        sns.histplot(residuals, kde=True, color=PURPLE, bins=30, ax=ax3,
                     edgecolor="none", alpha=0.7)
        ax3.axvline(x=0, color=DANGER, linestyle="--", linewidth=1.4, label="Error = 0")
        ax3.set_title("Distribusi Residual (Actual − Predicted)")
        ax3.set_xlabel("Residual"); ax3.set_ylabel("Frekuensi")
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.4, axis="y")
        fig3.tight_layout()
        fig_to_st(fig3)

        section("Bar Chart Metrik Evaluasi")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        colors = [ACCENT, ACCENT2, SUCCESS, WARN]
        bars = ax4.bar(["MSE","RMSE","MAD","MAPE"], [mse, rmse, mad, mape],
                       color=colors, width=0.5, edgecolor="none", alpha=0.85)
        for bar, val in zip(bars, [mse, rmse, mad, mape]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=9.5, color="#475569")
        ax4.set_title("Evaluasi Stacking Regressor")
        ax4.grid(True, alpha=0.4, axis="y")
        fig4.tight_layout()
        fig_to_st(fig4)