# TA-07: Q-Learning untuk Robot Kurir Pengantar Paket

Implementasi Q-Learning tabular (Python + NumPy murni, tanpa Gymnasium) untuk skenario **Robot Kurir 5×5 dengan manajemen baterai**, lanjutan dari TP-07.

## Skenario singkat

- **Peta 5×5:** titik ambil **P** (0,0), tujuan **D** (4,4), stasiun pengisian **C** (4,0), 3 rintangan. Robot mulai di P dengan baterai penuh.
- **State (200):** posisi 25 × status paket 2 × level baterai 4.
- **Action (7):** atas, bawah, kiri, kanan, ambil paket, serahkan paket, isi baterai.
- **Reward:** −1 per aksi; +20 ambil paket; +100 serahkan paket; −10 aksi tidak valid/menabrak; −50 baterai habis.
- **Dinamika:** setiap perpindahan berhasil punya peluang 30% menurunkan baterai satu level.

## Struktur proyek

```
.
├── TA07_QLearning_Kurir.ipynb      # Langkah 1-5 lengkap (sudah berisi output)
├── kurir_rl.py                     # Lingkungan + Q-Learning + visualisasi (dipakai skrip eksperimen)
├── experiment_hyperparameter.py    # Skrip eksperimen γ (Langkah 5)
├── requirements.txt
├── q_table.npy / q_table.csv       # Q-Table hasil pelatihan (200 × 7)
├── results_gamma.csv               # Hasil skrip eksperimen
└── figures/                        # Peta lingkungan, kurva pembelajaran, peta kebijakan, eksperimen γ
```

## Cara menjalankan

```bash
pip install -r requirements.txt
jupyter notebook TA07_QLearning_Kurir.ipynb      # jalankan semua sel dari atas ke bawah
python experiment_hyperparameter.py              # opsional: ulangi eksperimen γ lewat skrip
```

Notebook bersifat mandiri (seluruh kode lingkungan dan Q-Learning ada di dalamnya), jadi di Google Colab cukup upload satu file notebook. Skrip `experiment_hyperparameter.py` membutuhkan `kurir_rl.py` di folder yang sama.

## Hasil utama (seed 42)

| | α=0.1, γ=0.9 | γ=0.1 (ekstrem) |
|---|---|---|
| Tingkat sukses (1000 episode uji) | 88.8% | 50.4% |
| Rata-rata reward uji | 91.7 | −454.7 |
| Konvergensi | episode ≈ 2322 | tidak konvergen |

Semua eksperimen memakai seed tetap sehingga dapat direproduksi. Angka bisa sedikit berbeda jika versi NumPy berbeda.
