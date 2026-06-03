import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Lama Rawat DBD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SESSION STATE ───────────────────────────────────────────────────────────
for key in ["df", "df_scaled", "model", "scaler", "le_kelamin", "le_demam",
            "X_test", "y_test", "y_pred", "features", "trained"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# ─── HELPERS ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes, filename):
    """Load CSV/XLS dataset and run full preprocessing pipeline."""
    try:
        if filename.endswith(".xls") or filename.endswith(".csv.xls"):
            df = pd.read_csv(file_bytes, sep="\t")
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_excel(file_bytes)
    except Exception:
        df = pd.read_csv(file_bytes, sep=",")

    # Drop unused columns
    junk = [
        "Unnamed: 9", "Unnamed: 10", "preprocessing",
        "Normalisasi, cleaning, missing value, duplikate data, balancing, transformasi",
        "Unnamed: 13",
    ]
    df.drop(columns=[c for c in junk if c in df.columns], inplace=True)

    # Missing values & duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Encode categoricals
    le_k = LabelEncoder()
    le_d = LabelEncoder()
    df["jenis_kelamin_enc"] = le_k.fit_transform(df["jenis_kelamin"])
    df["jenis_demam_enc"] = le_d.fit_transform(df["jenis_demam"])

    # Date → numeric
    df["tgl_masuk"] = pd.to_datetime(df["tgl_masuk"], dayfirst=True)
    df["tgl_keluar"] = pd.to_datetime(df["tgl_keluar"], dayfirst=True)
    min_date = df["tgl_masuk"].min()
    df["tgl_masuk_num"] = (df["tgl_masuk"] - min_date).dt.days
    df["tgl_keluar_num"] = (df["tgl_keluar"] - min_date).dt.days

    # Normalisasi
    scaler = MinMaxScaler()
    num_cols = ["umur", "hemoglobin", "hct", "trombosit", "tgl_masuk_num", "tgl_keluar_num"]
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    return df, df_scaled, scaler, le_k, le_d


def train_model(df_scaled):
    features = ["umur", "hemoglobin", "hct", "trombosit", "jenis_kelamin_enc", "jenis_demam_enc"]
    X = df_scaled[features]
    y = df_scaled["lama_dirawat"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_regressors = [
        ("lin_reg", LinearRegression()),
        ("knn_reg", KNeighborsRegressor(n_neighbors=5)),
        ("svm_reg", SVR(kernel="rbf")),
        ("ann_reg", MLPRegressor(hidden_layer_sizes=(100,), max_iter=4000, random_state=42)),
    ]
    meta = LinearRegression()
    model = StackingRegressor(estimators=base_regressors, final_estimator=meta, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, features, X_test, y_test, y_pred


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("🏥 DBD Predictor")
    st.markdown("**Kelompok 7 — KB A**")
    st.divider()

    nav = st.radio(
        "Navigasi",
        ["📂 Upload & Preprocessing", "📊 EDA", "🤖 Training Model",
         "🔮 Prediksi Pasien", "📈 Evaluasi Model"],
        index=0,
    )
    st.divider()
    st.caption("Dataset: Dengue Fever LOS\nTarget: `lama_dirawat` (hari)")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
if nav == "📂 Upload & Preprocessing":
    st.title("📂 Upload Dataset & Preprocessing")
    st.markdown(
        "Upload file dataset dengue fever. Format yang didukung: `.csv`, `.xls`, `.xlsx`."
    )

    uploaded = st.file_uploader(
        "Pilih file dataset", type=["csv", "xls", "xlsx"], label_visibility="collapsed"
    )

    if uploaded:
        with st.spinner("Memproses dataset..."):
            df, df_scaled, scaler, le_k, le_d = load_and_preprocess(
                uploaded, uploaded.name
            )
            st.session_state.df = df
            st.session_state.df_scaled = df_scaled
            st.session_state.scaler = scaler
            st.session_state.le_kelamin = le_k
            st.session_state.le_demam = le_d
            st.session_state.trained = False  # reset model on new upload

        st.success(f"✅ Dataset berhasil dimuat! **{df.shape[0]} baris, {df.shape[1]} kolom**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Baris", df.shape[0])
        col2.metric("Total Kolom", df.shape[1])
        col3.metric("Missing Value", df.isnull().sum().sum())

        st.subheader("5 Data Pertama")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Encoding Kategorikal")
        c1, c2 = st.columns(2)
        with c1:
            mapping_k = dict(zip(le_k.classes_, le_k.transform(le_k.classes_)))
            st.write("**Jenis Kelamin**", mapping_k)
        with c2:
            mapping_d = dict(zip(le_d.classes_, le_d.transform(le_d.classes_)))
            st.write("**Jenis Demam**", mapping_d)

        st.subheader("Data Setelah Normalisasi (5 Teratas)")
        num_cols = ["umur", "hemoglobin", "hct", "trombosit", "tgl_masuk_num", "tgl_keluar_num"]
        st.dataframe(df_scaled[num_cols].head(), use_container_width=True)
    else:
        st.info("⬆️ Silakan upload dataset untuk memulai.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset terlebih dahulu di menu **Upload & Preprocessing**.")
    else:
        df = st.session_state.df
        df_scaled = st.session_state.df_scaled

        # --- Distribusi data klinis
        st.subheader("Distribusi Data Klinis")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df["umur"], kde=True, color="skyblue", ax=axes[0])
        axes[0].set_title("Distribusi Umur")
        sns.histplot(df["trombosit"], kde=True, color="salmon", ax=axes[1])
        axes[1].set_title("Distribusi Trombosit")
        df["jenis_demam"].value_counts().plot(kind="bar", ax=axes[2], color="steelblue")
        axes[2].set_title("Frekuensi Jenis Demam")
        axes[2].tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Distribusi lama dirawat
        st.subheader("Distribusi Lama Dirawat (Target)")
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=df["lama_dirawat"], color="lightgreen", ax=axes2[0])
        axes2[0].set_title("Boxplot Lama Dirawat")
        sns.violinplot(x=df["lama_dirawat"], color="lightgreen", ax=axes2[1])
        axes2[1].set_title("Violin Plot Lama Dirawat")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # --- Rata-rata lama dirawat per jenis demam
        st.subheader("Rata-rata Lama Dirawat per Jenis Demam")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=df, x="jenis_demam", y="lama_dirawat",
            estimator=np.mean, hue="jenis_demam", palette="muted",
            legend=False, ax=ax3,
        )
        ax3.set_title("Rata-rata Lama Dirawat berdasarkan Jenis Demam")
        ax3.set_ylabel("Rata-rata Hari")
        st.pyplot(fig3)
        plt.close()

        # --- Heatmap korelasi
        st.subheader("Heatmap Korelasi Pearson")
        features = ["umur", "hemoglobin", "hct", "trombosit", "lama_dirawat"]
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_scaled[features].corr(), annot=True, cmap="RdYlGn", center=0, ax=ax4)
        ax4.set_title("Heatmap Korelasi Pearson")
        st.pyplot(fig4)
        plt.close()

        # --- Tabel korelasi
        st.subheader("Tabel Korelasi & Koefisien Determinasi")
        feat_num = ["umur", "hemoglobin", "hct", "trombosit"]
        target = "lama_dirawat"
        pearson = df_scaled[feat_num + [target]].corr(method="pearson")[target].drop(target)
        spearman = df_scaled[feat_num + [target]].corr(method="spearman")[target].drop(target)
        r_scores = {}
        for col in feat_num:
            m = LinearRegression()
            m.fit(df_scaled[[col]], df_scaled[target])
            r_scores[col] = m.score(df_scaled[[col]], df_scaled[target])

        results_df = pd.DataFrame({
            "Pearson": pearson, "Spearman": spearman,
            "R-Squared": pd.Series(r_scores),
        })
        st.dataframe(results_df.style.background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)

        # --- Chi-square
        contingency = pd.crosstab(df["jenis_demam"], df["lama_dirawat"])
        chi2, p, dof, _ = chi2_contingency(contingency)
        st.info(f"**Chi-Square (Jenis Demam vs Lama Dirawat):** Chi2 = `{chi2:.4f}`, p-value = `{p:.4f}`")

        # --- Silhouette
        X_clust = df_scaled[feat_num]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_clust)
        sil = silhouette_score(X_clust, kmeans.labels_)
        st.info(f"**Silhouette Coefficient** (K-Means, k=3): `{sil:.4f}`")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🤖 Training Model":
    st.title("🤖 Training Stacking Regressor")

    if st.session_state.df_scaled is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        st.markdown("""
        **Arsitektur Model:**
        - 🔵 Base Regressors: Linear Regression, KNN (k=5), SVR (RBF), MLP (100 neurons)
        - 🔴 Meta Regressor: Linear Regression
        - 🔁 Cross-validation: 5-fold
        """)

        if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
            with st.spinner("Sedang melatih model Stacking Regressor... (bisa beberapa menit)"):
                model, features, X_test, y_test, y_pred = train_model(
                    st.session_state.df_scaled
                )
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.trained = True

            st.success("✅ Training selesai!")

        if st.session_state.trained:
            st.subheader("Fitur yang Digunakan")
            st.write(st.session_state.features)

            st.subheader("Ukuran Data")
            df_s = st.session_state.df_scaled
            n_train = int(len(df_s) * 0.8)
            n_test = len(df_s) - n_train
            col1, col2 = st.columns(2)
            col1.metric("Training Set", n_train)
            col2.metric("Test Set", n_test)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Prediksi Pasien":
    st.title("🔮 Prediksi Lama Rawat Pasien Baru")

    if not st.session_state.trained:
        st.warning("⚠️ Latih model terlebih dahulu di menu **Training Model**.")
    else:
        le_k = st.session_state.le_kelamin
        le_d = st.session_state.le_demam
        scaler = st.session_state.scaler
        model = st.session_state.model
        df = st.session_state.df

        st.markdown("Masukkan data pasien di bawah ini:")

        with st.form("form_prediksi"):
            col1, col2 = st.columns(2)
            with col1:
                jenis_kelamin = st.selectbox("Jenis Kelamin", list(le_k.classes_))
                umur = st.number_input("Umur (tahun)", min_value=0, max_value=120, value=25)
                jenis_demam = st.selectbox("Jenis Demam", list(le_d.classes_))
            with col2:
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.0, step=0.1)
                hct = st.number_input("Hematokrit / HCT (%)", min_value=0.0, max_value=70.0, value=40.0, step=0.1)
                trombosit = st.number_input("Trombosit (×10³/µL)", min_value=0, max_value=1000, value=150)

            submitted = st.form_submit_button("🔮 Prediksi", type="primary", use_container_width=True)

        if submitted:
            # Encode kategoris
            kel_enc = le_k.transform([jenis_kelamin])[0]
            dem_enc = le_d.transform([jenis_demam])[0]

            # Normalisasi numerik (pakai batas dari data asli)
            num_cols = ["umur", "hemoglobin", "hct", "trombosit", "tgl_masuk_num", "tgl_keluar_num"]
            # Buat array dummy (tgl_masuk_num & tgl_keluar_num = 0, dummy)
            raw_input = np.array([[umur, hemoglobin, hct, trombosit, 0, 0]], dtype=float)
            scaled_input = scaler.transform(raw_input)
            umur_s, hemo_s, hct_s, trom_s = scaled_input[0][:4]

            input_arr = np.array([[umur_s, hemo_s, hct_s, trom_s, kel_enc, dem_enc]])
            y_pred_raw = model.predict(input_arr)[0]

            # Denormalisasi target (MinMaxScaler hanya pada fitur numerik, bukan target)
            # Target lama_dirawat tidak di-scale, jadi hasil langsung dalam hari
            y_pred_days = round(y_pred_raw)

            st.divider()
            st.subheader("Hasil Prediksi")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("🛏️ Estimasi Lama Dirawat", f"{y_pred_days} hari", delta=None)
                st.metric("Nilai Prediksi (scaled)", f"{y_pred_raw:.4f}")
            with col_b:
                st.info(
                    f"Berdasarkan data pasien:\n"
                    f"- Jenis Kelamin: **{jenis_kelamin}**\n"
                    f"- Umur: **{umur} tahun**\n"
                    f"- Jenis Demam: **{jenis_demam}**\n"
                    f"- Hemoglobin: **{hemoglobin} g/dL**\n"
                    f"- Hematokrit: **{hct}%**\n"
                    f"- Trombosit: **{trombosit}×10³/µL**\n\n"
                    f"Model memperkirakan pasien akan dirawat selama **±{y_pred_days} hari**."
                )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — EVALUASI
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Stacking Regressor")

    if not st.session_state.trained:
        st.warning("⚠️ Latih model terlebih dahulu di menu **Training Model**.")
    else:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mad = mean_absolute_error(y_test, y_pred)
        y_safe = np.where(y_test == 0, 1e-10, y_test)
        mape = mean_absolute_percentage_error(y_safe, y_pred)

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE", f"{mse:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("MAD", f"{mad:.4f}")
        c4.metric("MAPE", f"{mape:.4f}")

        st.divider()
        col_l, col_r = st.columns(2)

        # --- Bar chart evaluasi
        with col_l:
            st.subheader("Bar Chart Metrik Evaluasi")
            fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
            ax_bar.bar(["MSE", "RMSE", "MAD", "MAPE"], [mse, rmse, mad, mape],
                       color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
            ax_bar.set_title("Evaluasi Stacking Regressor")
            ax_bar.set_ylabel("Value")
            st.pyplot(fig_bar)
            plt.close()

        # --- Actual vs Prediction
        with col_r:
            st.subheader("Actual vs Prediction")
            fig_ap, ax_ap = plt.subplots(figsize=(7, 4))
            ax_ap.plot(y_test.values, label="Actual", alpha=0.8)
            ax_ap.plot(y_pred, label="Prediction", alpha=0.8)
            ax_ap.set_title("Actual vs Prediction")
            ax_ap.set_xlabel("Data Index")
            ax_ap.set_ylabel("Nilai")
            ax_ap.legend()
            st.pyplot(fig_ap)
            plt.close()

        # --- Residual histogram
        st.subheader("Distribusi Residual (Error Prediksi)")
        residuals = y_test.values - y_pred
        fig_res, ax_res = plt.subplots(figsize=(10, 4))
        sns.histplot(residuals, kde=True, color="purple", bins=30, ax=ax_res)
        ax_res.axvline(x=0, color="red", linestyle="--", label="Error = 0")
        ax_res.set_title("Distribusi Residual")
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.set_ylabel("Frekuensi")
        ax_res.legend()
        ax_res.grid(True, alpha=0.3)
        st.pyplot(fig_res)
        plt.close()

        # --- Scatter actual vs predicted
        st.subheader("Scatter Plot: Actual vs Predicted")
        fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
        ax_sc.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax_sc.plot([mn, mx], [mn, mx], "r--", label="Perfect Fit")
        ax_sc.set_xlabel("Actual")
        ax_sc.set_ylabel("Predicted")
        ax_sc.set_title("Actual vs Predicted (Scatter)")
        ax_sc.legend()
        st.pyplot(fig_sc)
        plt.close()
