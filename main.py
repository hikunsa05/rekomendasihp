import streamlit as st
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# Membaca data dari file Excel
data = pd.read_excel('dataset/data hp.xlsx')

# Pastikan data dimuat dengan benar
st.write("### Data yang dimuat:")
st.dataframe(data) 

# Streamlit UI
st.title("Rekomendasi Smartphone")
st.write("Masukkan preferensi Anda untuk mendapatkan rekomendasi smartphone:")

# Input pengguna
input_name = st.text_input("Masukkan nama hp yang diinginkan").lower()
input_rom = st.slider("Masukkan kapasitas ROM yang diinginkan (dalam GB)", 8, 512, 64)  # Slider untuk ROM
input_ram = st.slider("Masukkan kapasitas RAM yang diinginkan (dalam GB)", 2, 16, 4)  # Slider untuk RAM

# Cek apakah semua input sudah diisi
if input_name and input_rom and input_ram:
    # Content-Based Filtering
    # TF-IDF Vectorization untuk nama smartphone
    tfidf = TfidfVectorizer(stop_words='english')  # Mengabaikan kata-kata umum (stopwords)
    name_vectors = tfidf.fit_transform(data['Name'].str.lower())

    # Gabungkan fitur TF-IDF dengan ROM, RAM, dan Ratings
    numerical_features = data[['ROM(GB)', 'RAM(GB)', 'Ratings']].values
    combined_features = pd.concat([
        pd.DataFrame(name_vectors.toarray()),
        pd.DataFrame(numerical_features)
    ], axis=1).values

    # Query pengguna
    query_vector = tfidf.transform([input_name]).toarray()
    query_numerical = [[input_rom, input_ram, 0]]  # Ratings diatur ke 0 untuk pencarian
    query_combined = pd.concat([
        pd.DataFrame(query_vector),
        pd.DataFrame(query_numerical)
    ], axis=1).values

    # Hitung kesamaan menggunakan cosine similarity
    similarities = cosine_similarity(query_combined, combined_features)

    # Tambahkan skor kesamaan ke dataset
    data['Similarity'] = similarities.flatten()

    # Filter hasil berdasarkan nama, ROM, dan RAM
    data_filtered = data[
        (data['Name'].str.lower().str.contains(input_name)) &  # Filter nama berbasis input
        (data['ROM(GB)'] == input_rom) &
        (data['RAM(GB)'] == input_ram)
    ]

    # Periksa apakah ada data yang cocok
    if not data_filtered.empty:
        # Urutkan hasil berdasarkan kesamaan tertinggi
        recommended = data_filtered.sort_values(by='Similarity', ascending=False)

        # Tampilkan hasil dalam format tabel yang rapi menggunakan st.dataframe
        st.write("### Hasil Rekomendasi:")
        st.dataframe(recommended[['Name', 'ROM(GB)', 'RAM(GB)', 'Ratings', 'Price', 'Similarity']])
    else:
        st.write("Tidak ada smartphone yang memenuhi kriteria pencarian Anda.")
else:
    st.write("Silakan masukkan semua preferensi untuk mendapatkan rekomendasi.")
    

st.write("")
st.write("")
# HISTOGRAM
# Membuat kolom 'Type' dari data 'Name'
data['Type'] = data['Name'].str.split().str[0]

# Menghitung jumlah data berdasarkan 'Type' dan 'Ratings'
type_ratings_counts = data.groupby(['Type', 'Ratings']).size().reset_index(name='Count')

# Membuat palet warna
colors = sns.color_palette("viridis", len(type_ratings_counts))

# Membuat plot dengan Matplotlib
st.write("### Grafik Jumlah HP Berdasarkan Tipe dan Rating")

plt.figure(figsize=(20, 7))  # Ukuran plot
bars = plt.bar(type_ratings_counts['Type'], type_ratings_counts['Count'], color=colors)

# Menambahkan judul dan label
plt.title('Jumlah HP Berdasarkan Tipe dan Rating', fontsize=14)
plt.xlabel('Tipe HP', fontsize=12)
plt.ylabel('Jumlah HP', fontsize=12)
plt.xticks(rotation=90)

# Menambahkan teks pada setiap bar
for bar, rating, count in zip(bars, type_ratings_counts['Ratings'], type_ratings_counts['Count']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{rating}\n({count})",
        ha='center',
        va='bottom',
        fontsize=8
    )

# Layout untuk memastikan semuanya sesuai
plt.tight_layout()

# Menampilkan grafik ke dalam Streamlit
st.pyplot(plt)

st.markdown("""
Insight:
- Realme memiliki jumlah HP terbanyak dengan angka mencapai 40 unit dan rating rata-rata 4.4, menunjukkan popularitas produk mereka yang konsisten dengan penilaian yang cukup baik.
- Merek seperti POCO dan OPPO mencapai rating tertinggi di angka 4.5, meskipun jumlah produknya lebih sedikit dibandingkan Realme. Ini menunjukkan kualitas produk mereka diapresiasi pengguna atau memenuhi kebutuhan pengguna.
""")


st.write("")
st.write("")
# BOXPLOT
# Filter 10 HP dengan harga tertinggi
top_price_hp = data.nlargest(10, 'Price')

# Membuat plot boxplot
st.write("### Boxplot 10 HP dengan Harga Tertinggi")

plt.figure(figsize=(14, 6))
sns.boxplot(x='Name', y='Price', data=top_price_hp)
plt.xticks(rotation=90)
plt.title('Boxplot 10 HP dengan Harga Tertinggi')
plt.ylabel('Price')
plt.xlabel('Nama HP')

# Menampilkan plot ke Streamlit
st.pyplot(plt)

st.markdown("""
Insight:
- Boxplot memperlihatkan variasi yang signifikan dalam data hp dengan berbagai harga.
- Beberapa tipe menunjukkan harga yang lebih tinggi sesuai dengan spesifikasi hp.
""")


st.write("")
st.write("")
# SCATTER PLOT
# Menampilkan scatter plot di Streamlit
st.write("### Scatter Plot: Ratings vs Price")

plt.figure(figsize=(10, 6))
plt.scatter(data['Ratings'], data['Price'], alpha=0.5, color='green')
plt.title('Total Ratings vs Price')
plt.xlabel('Ratings')
plt.ylabel('Price')

# Menampilkan plot di Streamlit
st.pyplot(plt)

st.markdown("""
Insight:
- Scatter plot tersebut menunjukkan hubungan positif antara kapasitas rating dan harga. 
- HP dengan harga tinggi tidak selalu memiliki rating tertinggi, dan begitu pula sebaliknya.
- Hal ini juga bisa disebabkan oleh faktor lain seperti spesifikasi ROM, RAM dan fitur tambahan juga bisa mempengaruhi harga..
""")


st.write("")
st.write("")
# HEATMAP
# Pilih kolom numerik untuk korelasi
numerical_columns = ['Ratings', 'Price', 'ROM(GB)', 'RAM(GB)']  # Ganti dengan kolom numerik yang relevan di dataset Anda

# Periksa apakah kolom numerik ada di data
data_numerical = data[numerical_columns].dropna()  # Menghapus baris dengan nilai NaN

# Menampilkan heatmap di Streamlit
st.write("### Heatmap Korelasi Data HP")

plt.figure(figsize=(12, 8))
correlation_matrix = data_numerical.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Data HP')

# Tampilkan heatmap di Streamlit
st.pyplot(plt)

st.markdown("""
Insight:
- ROM dan RAM memiliki korelasi yang positif. Dimana, HP dengan kapasitas ROM yang besar cenderung memiliki kapasitas RAM yang besar juga.
- RAM, ROM dan harga memiliki korelasi yang sangat kuat. Semakin besar RAM, semakin mahal harga HP. 
- Terdapat korelasi positif rendah antara rating pengguna dan harga. Ini menunjukkan bahwa harga lebih tinggi tidak selalu menghasilkan ulasan pengguna yang baik, karena ulasan pengguna dapat dipengaruhi oleh faktor lain.
""")


st.write("")
st.write("")
# LINE PLOT 
# Mengelompokkan data berdasarkan Harga
data_grouped = data.groupby('Price').sum()

# Streamlit UI
st.write("### Line Plot: Jumlah Rating HP Berdasarkan Harga")

# Membuat plot
plt.figure(figsize=(10, 6))
plt.plot(data_grouped.index, data_grouped['Ratings'], label='Rating HP', color='purple')
plt.title('Jumlah Rating HP Berdasarkan Harga')
plt.xlabel('Harga')
plt.ylabel('Jumlah Rating')
plt.xticks(rotation=45)
plt.legend()

# Tampilkan plot di Streamlit
st.pyplot(plt)

st.markdown("""
Insight:
- Grafik menunjukkan bahwa jumlah rating tidak merata di berbagai rentang harga.
- Puncak jumlah rating terjadi di harga tertentu, khususnya pada rentang harga sekitar 10000 hingga 15000.
- Produk dengan harga sekitar 10.000 mendapatkan lebih banyak perhatian dari konsumen, yang terlihat dari tingginya jumlah rating.
- Di rentang harga yang lebih tinggi (di atas 20.000), jumlah rating cenderung menurun, menunjukkan bahwa produk di harga ini mungkin memiliki pasar yang lebih kecil atau daya tarik yang lebih rendah.
""")
