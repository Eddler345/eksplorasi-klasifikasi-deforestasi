# Eksplorasi Klasifikasi Deforestasi Menggunakan Data Geospasial

## Ringkasan

Proyek pembelajaran ini mengeksplorasi penggunaan jaringan saraf tiruan (*deep learning*) untuk mengelompokkan tiga kategori penyebab deforestasi: pertanian, pertambangan, dan penebangan. Program dibuat menggunakan Python, TensorFlow/Keras, NumPy, Pandas, dan Scikit-learn.

**Catatan penting:** versi ini menggunakan data sintetis yang dibangkitkan secara acak untuk mendemonstrasikan alur kerja model. Data tersebut bukan citra Sentinel-1 atau Sentinel-2 asli, sehingga hasil evaluasinya tidak dapat dianggap sebagai ukuran akurasi untuk kondisi deforestasi nyata.

## Alur program

1. Membuat data sintetis dengan 36 fitur dan tiga label kelas.
2. Membagi data menjadi bagian pelatihan dan pengujian.
3. Membentuk model jaringan saraf dengan dua lapisan tersembunyi dan *dropout*.
4. Melatih model dan menghasilkan prediksi untuk data pengujian.
5. Menampilkan laporan klasifikasi serta menyimpan hasil prediksi ke file CSV.

## Struktur repositori

```text
eksplorasi-klasifikasi-deforestasi/
├── main.py
├── requirements.txt
└── README.md
```

## Persyaratan

- Python 3.10 atau versi yang kompatibel dengan versi TensorFlow yang dipasang.
- Paket Python yang tercantum di `requirements.txt`.

## Instalasi dan menjalankan

Buat lingkungan virtual jika diperlukan, lalu jalankan:

```bash
python -m pip install -r requirements.txt
python main.py
```

Setelah program selesai, hasil prediksi akan disimpan sebagai `hasil_klasifikasi_deforestasi.csv` di direktori kerja saat ini.

## Batasan dan pengembangan lanjutan

- Data yang digunakan saat ini adalah data acak untuk demonstrasi alur, bukan data penginderaan jauh nyata.
- Karena label dibuat secara acak dan tidak dibentuk berdasarkan pola fitur, metrik model tidak menunjukkan kemampuan mengenali penyebab deforestasi yang sebenarnya.
- Pengembangan berikutnya dapat memakai dataset geospasial berlabel yang dapat diverifikasi, mendokumentasikan sumber dan lisensinya, serta mengevaluasi model dengan pembagian data yang sesuai.

## Teknologi

- Python
- NumPy dan Pandas
- TensorFlow/Keras
- Scikit-learn
