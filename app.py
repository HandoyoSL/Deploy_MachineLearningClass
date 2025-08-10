import os, json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Income", page_icon="🤖", layout="centered")
st.title("🔮 Prediksi Income (Adult)")

# --- Load schema dari file ---
with open("feature_schema.json", "r", encoding="utf-8") as f:
    EXPECTED_COLUMNS = json.load(f)["columns"]

# --- Load model (.joblib prefer, fallback .pkl) ---
@st.cache_resource
def load_model():
    path = "model.joblib" if os.path.exists("model.joblib") else "model.pkl"
    if path.endswith(".joblib"):
        from joblib import load
        return load(path)
    else:
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)

model = load_model()

# ==== Pilihan kategori (samakan ejaannya dgn training data!) ====
OCCUPATIONS = [
    "Tech-support","Craft-repair","Other-service","Sales","Exec-managerial","Prof-specialty",
    "Handlers-cleaners","Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving",
    "Priv-house-serv","Protective-serv","Armed-Forces"
]
RACES = ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"]
GENDERS = ["Male","Female"]
COUNTRIES = [
    "United-States","Mexico","Philippines","Germany","Canada","Puerto-Rico","El-Salvador","India","Cuba",
    "England","Jamaica","South","China","Italy","Dominican-Republic","Vietnam","Guatemala","Japan","Poland",
    "Columbia","Taiwan","Haiti","Portugal","Iran","Nicaragua","Peru","Greece","France","Ecuador","Ireland",
    "Hong","Cambodia","Trinadad&Tobago","Laos","Thailand","Yugoslavia","Outlying-US(Guam-USVI-etc)",
    "Honduras","Scotland","Holand-Netherlands"
]

st.caption("Pastikan nama & ejaan kategori sama persis dengan data training (OneHotEncoder sebaiknya handle_unknown='ignore').")

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=EXPECTED_COLUMNS)

# ===== UI =====
mode = st.radio("Pilih mode input", ["Form", "Batch CSV"], horizontal=True)

if mode == "Form":
    st.subheader("Isi Fitur")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("age", min_value=0, max_value=100, value= thirty if (thirty:=30) else 30, step=1)
        edu_num = st.number_input("educational-num", min_value=1, max_value=16, value=9, step=1)
        capital_gain = st.number_input("capital-gain", min_value=0, max_value=99999, value=0, step=1)
        capital_loss = st.number_input("capital-loss", min_value=0, max_value=99999, value=0, step=1)
        hours_per_week = st.number_input("hours-per-week", min_value=1, max_value=99, value=40, step=1)
    with col2:
        occupation = st.selectbox("occupation", OCCUPATIONS, index=OCCUPATIONS.index("Adm-clerical") if "Adm-clerical" in OCCUPATIONS else 0)
        race = st.selectbox("race", RACES, index=RACES.index("White") if "White" in RACES else 0)
        gender = st.selectbox("gender", GENDERS, index=0)
        native_country = st.selectbox("native-country", COUNTRIES, index=COUNTRIES.index("United-States") if "United-States" in COUNTRIES else 0)

    row = {
        "age": age,
        "educational-num": edu_num,
        "occupation": occupation,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }

    df = pd.DataFrame([row])
    df = ensure_schema(df)

    if st.button("Prediksi"):
        try:
            pred = model.predict(df)[0]
            st.success(f"Prediksi: {pred}")
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df)[0]
                    classes = getattr(model, "classes_", list(range(len(proba))))
                    st.write("Probabilitas kelas:", {str(c): float(p) for c, p in zip(classes, proba)})
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.subheader("Upload CSV untuk prediksi batch")
    st.caption("CSV harus berisi kolom persis seperti schema (tanpa kolom target).")
    file = st.file_uploader("Unggah CSV", type=["csv"])
    if file:
        try:
            df_csv = pd.read_csv(file)
            df_csv = ensure_schema(df_csv)
            st.write("Preview:")
            st.dataframe(df_csv.head())

            if st.button("Prediksi CSV"):
                preds = model.predict(df_csv)
                out = pd.DataFrame({"prediction": preds})
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(df_csv)
                        if proba.shape[1] == 2:
                            out["proba_positive"] = proba[:, 1]
                        else:
                            classes = getattr(model, "classes_", list(range(proba.shape[1])))
                            for i, c in enumerate(classes):
                                out[f"proba_{c}"] = proba[:, i]
                    except Exception:
                        pass
                st.success("Selesai!")
                st.dataframe(out.head())
                st.download_button("Unduh predictions.csv", out.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Error: {e}")

with st.expander("Debug"):
    st.write("Tipe model:", type(model).__name__)
    if hasattr(model, "classes_"):
        st.write("Classes:", getattr(model, "classes_"))
    st.write("Schema:", EXPECTED_COLUMNS)