import streamlit as st
import pandas as pd
import joblib

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Prediksi Harga Motor",
    page_icon="🏍️",
    layout="centered"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("model_linear_regression.pkl")

df = pd.read_csv("D:\PrediksiHargaMotor\dataset_harga_motor.csv")

nama_motor = sorted(df["name"].unique())

# ==========================
# CSS
# ==========================
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #1e3a8a;
}

.sub {
    text-align: center;
    color: #64748b;
    margin-bottom: 25px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 5px 15px rgba(0,0,0,.1);
    margin-top: 20px;
}

.prediksi {
    font-size: 40px;
    color: green;
    text-align: center;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/741/741407.png",
        width=120
    )

    st.title("Informasi Model")

    st.success("Linear Regression")

    st.info("""
Dataset menggunakan fitur:

• Nama Motor

• Tahun

• Tipe Penjual

• Kepemilikan

• Kilometer

• Harga Baru (Ex Showroom)
""")

    st.write("---")

    st.caption("Deployment Machine Learning")

# ==========================
# Judul
# ==========================
st.markdown(
    '<p class="title">🏍️ Prediksi Harga Motor Bekas</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub">Prediksi harga jual motor menggunakan algoritma Linear Regression</p>',
    unsafe_allow_html=True
)

# ==========================
# Form Input
# ==========================
col1, col2 = st.columns(2)

# Ambil daftar nama motor
nama_motor = sorted(df["name"].unique())

with col1:

    # Pilih nama motor
    name = st.selectbox(
        "Nama Motor",
        nama_motor
    )

    # Ambil data motor yang dipilih
    motor = df[df["name"] == name].iloc[0]

    # Tahun otomatis mengikuti dataset
    year = st.number_input(
        "Tahun",
        min_value=2000,
        max_value=2035,
        value=int(motor["year"])
    )

    seller_list = [
        "Individual",
        "Dealer",
        "Trustmark Dealer"
    ]

    seller_type = st.selectbox(
        "Tipe Penjual",
        seller_list,
        index=seller_list.index(motor["seller_type"])
    )

with col2:

    owner_list = sorted(df["owner"].unique())

    owner = st.selectbox(
        "Kepemilikan",
        owner_list,
        index=owner_list.index(motor["owner"])
    )

    km_driven = st.number_input(
        "Kilometer",
        min_value=0,
        value=int(motor["km_driven"])
    )

    ex_showroom_price = st.number_input(
        "Harga Baru (Ex Showroom)",
        min_value=0.0,
        value=float(motor["ex_showroom_price"])
    )

st.write("")

prediksi = st.button(
    "🔍 Prediksi Harga",
    use_container_width=True
)

# ==========================
# Prediksi
# ==========================
if prediksi:

    data = pd.DataFrame({
        "name": [name],
        "year": [year],
        "seller_type": [seller_type],
        "owner": [owner],
        "km_driven": [km_driven],
        "ex_showroom_price": [ex_showroom_price]
    })

    hasil = model.predict(data)[0]

    # Kurs INR ke IDR
    kurs = 189
    harga_rupiah = hasil * kurs

    st.markdown(
        f"""
        <div class="card">

        <h2 align="center">Estimasi Harga Motor</h2>

        <p class="prediksi">
        ₹ {hasil:,.0f}
        </p>

        <h3 align="center">
        ≈ Rp {harga_rupiah:,.0f}
        </h3>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        f"Prediksi berhasil! Harga motor diperkirakan sekitar ₹ {hasil:,.0f} atau setara Rp {harga_rupiah:,.0f}"
    )

# ==========================
# Footer
# ==========================
st.markdown(
"""
<div class="footer">

Dibuat menggunakan ❤️ Streamlit | Scikit-Learn | Linear Regression

</div>
""",
unsafe_allow_html=True
)