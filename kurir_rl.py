"""Lingkungan Robot Kurir 5x5 dan Q-Learning tabular (TA-07).

Skenario ini adalah kelanjutan langsung dari TP-07. Semua logika memakai Python + NumPy murni
(tanpa Gymnasium). Matplotlib hanya dipakai oleh fungsi visualisasi.
"""
import numpy as np

# ---------------------------------------------------------------- Konfigurasi dunia
GRID = 5
P_CELL = (0, 0)                      # titik ambil paket
D_CELL = (4, 4)                      # titik tujuan
C_CELL = (4, 0)                      # stasiun pengisian baterai
START = P_CELL                       # robot memulai di titik ambil dengan baterai penuh
OBSTACLES = {(1, 1), (2, 3), (3, 1)}

ACTIONS = ["Atas", "Bawah", "Kiri", "Kanan", "Ambil Paket", "Serahkan Paket", "Isi Baterai"]
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
N_ACTIONS = len(ACTIONS)             # 7
N_BATTERY = 4                        # 0 = habis, 1 = rendah, 2 = sedang, 3 = penuh
N_STATES = GRID * GRID * 2 * N_BATTERY   # 25 x 2 x 4 = 200

# Reward (dijumlahkan: biaya langkah selalu berlaku, bonus/hukuman ditambahkan di atasnya)
R_STEP = -1
R_PICKUP = 20
R_DELIVER = 100
R_INVALID = -10
R_DEAD = -50
P_DRAIN = 0.3                        # peluang baterai turun 1 level saat robot berpindah


# ---------------------------------------------------------------- Encoding state
def encode(state):
    """(baris, kolom, status_paket, level_baterai) -> indeks 0..199."""
    r, c, pkg, bat = state
    return ((r * GRID + c) * 2 + pkg) * N_BATTERY + bat


def decode(idx):
    bat = idx % N_BATTERY
    idx //= N_BATTERY
    pkg = idx % 2
    idx //= 2
    return (idx // GRID, idx % GRID, pkg, bat)


# ---------------------------------------------------------------- Lingkungan
class KurirEnv:
    """Menerima (state, action) dan mengembalikan (new_state, reward, done, event)."""

    def __init__(self, seed=None, p_drain=P_DRAIN):
        self.rng = np.random.default_rng(seed)
        self.p_drain = p_drain

    def reset(self):
        return (START[0], START[1], 0, 3)

    def step(self, state, action):
        r, c, pkg, bat = state
        reward, done, event = R_STEP, False, None

        if action in MOVES:
            nr, nc = r + MOVES[action][0], c + MOVES[action][1]
            if not (0 <= nr < GRID and 0 <= nc < GRID) or (nr, nc) in OBSTACLES:
                reward += R_INVALID                      # menabrak rintangan / tepi grid
            else:
                r, c = nr, nc
                if self.rng.random() < self.p_drain:     # baterai turun secara acak
                    bat -= 1
                    if bat == 0:
                        reward += R_DEAD
                        done, event = True, "battery_dead"
        elif action == 4:                                # Ambil Paket
            if (r, c) == P_CELL and pkg == 0:
                pkg = 1
                reward += R_PICKUP
            else:
                reward += R_INVALID
        elif action == 5:                                # Serahkan Paket
            if (r, c) == D_CELL and pkg == 1:
                reward += R_DELIVER
                done, event = True, "delivered"
            else:
                reward += R_INVALID
        else:                                            # Isi Baterai
            if (r, c) == C_CELL:
                bat = 3
            else:
                reward += R_INVALID
        return (r, c, pkg, bat), reward, done, event


# ---------------------------------------------------------------- Q-Learning
def _greedy(q_row, rng):
    """Aksi terbaik; jika seri, pilih acak (menghindari bias ke aksi pertama)."""
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def train_q_learning(alpha=0.8, gamma=0.9, episodes=6000, max_steps=100,
                     eps_start=1.0, eps_min=0.01, decay_rate=0.001,
                     seed=42, p_drain=P_DRAIN):
    """Melatih agen. Mengembalikan (q_table, history)."""
    env = KurirEnv(seed=seed, p_drain=p_drain)
    rng = np.random.default_rng(seed + 1)
    q = np.zeros((N_STATES, N_ACTIONS))
    rewards, successes, epsilons = [], [], []

    for ep in range(episodes):
        eps = eps_min + (eps_start - eps_min) * np.exp(-decay_rate * ep)
        state, total, outcome = env.reset(), 0.0, None
        for _ in range(max_steps):
            s = encode(state)
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else _greedy(q[s], rng)   # epsilon-greedy
            new_state, r, done, event = env.step(state, a)
            ns = encode(new_state)
            target = r if done else r + gamma * q[ns].max()          # Persamaan Bellman
            q[s, a] = (1 - alpha) * q[s, a] + alpha * target
            total += r
            state = new_state
            if done:
                outcome = event
                break
        rewards.append(total)
        successes.append(1 if outcome == "delivered" else 0)
        epsilons.append(eps)

    return q, {"rewards": np.array(rewards), "successes": np.array(successes), "epsilons": np.array(epsilons)}


def evaluate_policy(q, n_episodes=1000, max_steps=100, seed=123, p_drain=P_DRAIN):
    """Mengikuti kebijakan greedy (tanpa eksplorasi) pada lingkungan baru."""
    env = KurirEnv(seed=seed, p_drain=p_drain)
    rewards, steps_ok, outcomes = [], [], {"delivered": 0, "battery_dead": 0, "timeout": 0}
    for _ in range(n_episodes):
        state, total, outcome = env.reset(), 0.0, "timeout"
        for t in range(1, max_steps + 1):
            a = int(np.argmax(q[encode(state)]))
            state, r, done, event = env.step(state, a)
            total += r
            if done:
                outcome = event
                if event == "delivered":
                    steps_ok.append(t)
                break
        rewards.append(total)
        outcomes[outcome] += 1
    return {
        "Tingkat sukses": outcomes["delivered"] / n_episodes,
        "Baterai habis": outcomes["battery_dead"] / n_episodes,
        "Timeout": outcomes["timeout"] / n_episodes,
        "Rata-rata reward": float(np.mean(rewards)),
        "Rata-rata langkah (sukses)": float(np.mean(steps_ok)) if steps_ok else float("nan"),
    }


def rollout(q, seed=7, max_steps=40, p_drain=P_DRAIN):
    """Satu episode greedy, dicatat langkah demi langkah (untuk demo / video)."""
    env = KurirEnv(seed=seed, p_drain=p_drain)
    state, log = env.reset(), []
    for t in range(1, max_steps + 1):
        a = int(np.argmax(q[encode(state)]))
        new_state, r, done, event = env.step(state, a)
        log.append({"Langkah": t, "State (baris,kolom,paket,baterai)": state, "Aksi": ACTIONS[a],
                    "Reward": r, "State berikutnya": new_state})
        state = new_state
        if done:
            break
    return log


def greedy_visit_counts(q, n_episodes=2000, max_steps=100, seed=321, p_drain=P_DRAIN):
    """Berapa kali tiap state dikunjungi saat mengikuti kebijakan greedy (untuk menandai state yang relevan)."""
    env = KurirEnv(seed=seed, p_drain=p_drain)
    counts = np.zeros(N_STATES, dtype=int)
    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            counts[encode(state)] += 1
            state, _, done, _ = env.step(state, int(np.argmax(q[encode(state)])))
            if done:
                break
    return counts


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def convergence_episode(rewards, window=100, frac=0.9, tail=500):
    """Episode pertama saat rata-rata bergerak mencapai `frac` dari plateau (rata-rata `tail` episode terakhir).

    Mengembalikan (None, plateau) jika plateau tidak positif, artinya agen tidak konvergen ke kebijakan yang baik.
    """
    ma = moving_average(rewards, window)
    plateau = float(np.mean(rewards[-tail:]))
    if plateau <= 0:
        return None, plateau
    hit = np.flatnonzero(ma >= frac * plateau)
    return (int(hit[0]) if len(hit) else None), plateau


# ---------------------------------------------------------------- Visualisasi
def plot_policy(q, pkg, bat, ax, vmin=None, vmax=None, title=None, visits=None):
    """Peta kebijakan 5x5 untuk status paket `pkg` dan level baterai `bat`.

    Jika `visits` diberikan, hanya state yang pernah dikunjungi kebijakan greedy yang digambar;
    state lain diredupkan (nilai Q-nya tidak relevan karena tidak pernah dilalui).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    symbols = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "ambil", 5: "antar", 6: "isi"}
    values = np.full((GRID, GRID), np.nan)
    relevant = np.zeros((GRID, GRID), dtype=bool)
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) in OBSTACLES:
                continue
            s_idx = encode((r, c, pkg, bat))
            relevant[r, c] = (visits is None and np.any(q[s_idx])) or (visits is not None and visits[s_idx] > 0)
            if relevant[r, c]:
                values[r, c] = q[s_idx].max()
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f1f1")
    im = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi)
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) in OBSTACLES:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#8a8a8a"))
                ax.text(c, r, "#", ha="center", va="center", fontsize=16, color="white")
            elif not relevant[r, c]:
                ax.text(c, r, "·", ha="center", va="center", fontsize=14, color="#bbbbbb")
            else:
                a = int(np.argmax(q[encode((r, c, pkg, bat))]))
                norm = (values[r, c] - lo) / max(1e-9, hi - lo)
                ax.text(c, r, symbols[a], ha="center", va="center", fontsize=18 if a < 4 else 9,
                        color="white" if norm < 0.6 else "black", fontweight="bold" if a >= 4 else "normal")
    for (r, c), tag in {P_CELL: "P", D_CELL: "D", C_CELL: "C"}.items():
        ax.text(c - 0.42, r - 0.32, tag, ha="left", va="center", fontsize=9, fontweight="bold", color="#e0a800")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=10)
    return im
