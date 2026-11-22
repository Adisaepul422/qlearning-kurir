"""Web app kecil untuk menampilkan hasil TA-07 (Q-Learning Robot Kurir) di Railway."""
import csv
import random
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import kurir_rl

BASE = Path(__file__).parent
Q = np.load(BASE / "q_table.npy")

app = FastAPI(title="TA-07 Q-Learning Robot Kurir")
app.mount("/figures", StaticFiles(directory=BASE / "figures"), name="figures")


def load_results():
    with open(BASE / "results_gamma.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/results")
def api_results():
    return load_results()


@app.get("/api/rollout")
def api_rollout(seed: int | None = None):
    """Jalankan satu episode dengan kebijakan greedy dari Q-Table hasil latihan."""
    seed = random.randint(0, 99999) if seed is None else seed
    log = kurir_rl.rollout(Q, seed=seed)
    steps = [
        {
            "langkah": row["Langkah"],
            "state": list(row["State (baris,kolom,paket,baterai)"]),
            "aksi": row["Aksi"],
            "reward": row["Reward"],
        }
        for row in log
    ]
    return {"seed": seed, "total_reward": sum(s["reward"] for s in steps), "steps": steps}


PAGE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TA-07 Q-Learning Robot Kurir</title>
<style>
  :root { --bg:#f6f7f9; --fg:#1b2430; --muted:#5d6b7c; --line:#d9dee5; --card:#fff; --accent:#1f5fbf; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12171d; --fg:#e8edf3; --muted:#9aa7b6; --line:#2a333d; --card:#1a2129; --accent:#6fa3ee; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 1.7rem; margin: 0 0 6px; }
  h2 { font-size: 1.15rem; margin: 40px 0 12px; }
  p { margin: 0 0 12px; max-width: 70ch; }
  .muted { color: var(--muted); }
  .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--card); }
  table { border-collapse: collapse; width: 100%; font-size: .92rem; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
  tr:last-child td { border-bottom: 0; }
  button { font: inherit; padding: 8px 16px; border-radius: 6px; border: 0; background: var(--accent); color: #fff; cursor: pointer; }
  button:focus-visible { outline: 3px solid var(--fg); outline-offset: 2px; }
  img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
  .figs { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
  figure { margin: 0; }
  figcaption { font-size: .85rem; color: var(--muted); margin-top: 4px; }
  .ok { color: #1d8a4c; } .bad { color: #c0392b; }
</style>
</head>
<body>
<main>
  <h1>Robot kurir belajar mengantar paket</h1>
  <p class="muted">TA-07: Q-Learning tabular pada peta 5×5 dengan baterai. Robot mengambil paket di P, mengisi baterai di C bila perlu, lalu mengantar ke D.</p>

  <h2>Coba satu episode</h2>
  <p>Robot memakai Q-Table hasil latihan dan selalu memilih aksi terbaik. Baterai turun secara acak, jadi tiap seed bisa berbeda.</p>
  <button id="run">Jalankan episode baru</button>
  <p id="summary" class="muted" style="margin-top:12px"></p>
  <div class="table-wrap" id="rollout-wrap" hidden>
    <table>
      <thead><tr><th>Langkah</th><th>Posisi (baris, kolom)</th><th>Paket</th><th>Baterai</th><th>Aksi</th><th>Reward</th></tr></thead>
      <tbody id="rollout"></tbody>
    </table>
  </div>

  <h2>Perbandingan γ</h2>
  <div class="table-wrap"><table id="results"></table></div>

  <h2>Gambar</h2>
  <div class="figs">
    <figure><img src="/figures/environment_map.png" alt="Peta lingkungan"><figcaption>Peta lingkungan</figcaption></figure>
    <figure><img src="/figures/learning_curve.png" alt="Kurva pembelajaran"><figcaption>Kurva pembelajaran</figcaption></figure>
    <figure><img src="/figures/policy_map.png" alt="Peta kebijakan"><figcaption>Peta kebijakan</figcaption></figure>
    <figure><img src="/figures/policy_gamma_compare.png" alt="Perbandingan kebijakan"><figcaption>Perbandingan kebijakan antar γ</figcaption></figure>
  </div>
</main>
<script>
async function loadResults() {
  const rows = await (await fetch('/api/results')).json();
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const t = document.getElementById('results');
  t.innerHTML = '<thead><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => '<td>' + r[c] + '</td>').join('') + '</tr>').join('') + '</tbody>';
}
async function runEpisode() {
  const data = await (await fetch('/api/rollout')).json();
  const last = data.steps[data.steps.length - 1];
  const delivered = last.aksi === 'Serahkan Paket' && last.reward > 0;
  document.getElementById('summary').innerHTML =
    'Seed ' + data.seed + ': ' + data.steps.length + ' langkah, total reward ' + data.total_reward + '. ' +
    (delivered ? '<span class="ok">Paket terantar.</span>' : '<span class="bad">Paket tidak terantar.</span>');
  document.getElementById('rollout').innerHTML = data.steps.map(s =>
    '<tr><td>' + s.langkah + '</td><td>(' + s.state[0] + ', ' + s.state[1] + ')</td><td>' + (s.state[2] ? 'dibawa' : 'belum') +
    '</td><td>' + s.state[3] + '/3</td><td>' + s.aksi + '</td><td>' + s.reward + '</td></tr>').join('');
  document.getElementById('rollout-wrap').hidden = false;
}
document.getElementById('run').addEventListener('click', runEpisode);
loadResults();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE