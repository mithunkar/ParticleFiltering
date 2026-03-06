"""
Particle Filter Inference for DBN with Interventions
Mirrors exact_inference.py exactly — same CPDs, same queries, side-by-side comparison.

DBN structure (binary variables, 0-indexed values matching exact_inference.py):
  W_t: Weather     0=sunny,   1=cloudy
  T_t: Temperature 0=hot,     1=cold
  E_t: Energy      0=high,    1=low
  U_t: Umbrella    0=no,      1=yes      (observed)
  A_t: Activity    0=outdoor, 1=indoor   (observed)

Dependencies:
  W_t | W_{t-1}
  T_t | T_{t-1}, W_{t-1}, W_t
  E_t | E_{t-1}, W_t, T_t
  U_t | W_t
  A_t | E_t, W_t
"""

import numpy as np
import time
from collections import defaultdict

Particle = dict   # varname -> int (0 or 1)

# ─────────────────────────────────────────────────────────────────────────────
# CPD tables — copied verbatim from exact_inference.py
# Key convention: tuple of parent values in the order listed, matching
# exact_inference.py's evidence= ordering.
# ─────────────────────────────────────────────────────────────────────────────

# P(W_0)
P_W0 = {(): [0.6, 0.4]}

# P(T_0 | W_0)
P_T0 = {
    (0,): [0.7, 0.3],   # sunny
    (1,): [0.3, 0.7],   # cloudy
}

# P(E_0 | W_0, T_0)
P_E0 = {
    (0, 0): [0.8, 0.2],  # sunny, hot
    (0, 1): [0.6, 0.4],  # sunny, cold
    (1, 0): [0.5, 0.5],  # cloudy, hot
    (1, 1): [0.3, 0.7],  # cloudy, cold
}

# P(U_t | W_t)  — same at all t
P_U = {
    (0,): [0.95, 0.05],  # sunny -> [no, yes]
    (1,): [0.20, 0.80],  # cloudy
}

# P(A_t | E_t, W_t)  — same at all t
P_A = {
    (0, 0): [0.9, 0.1],  # high, sunny
    (0, 1): [0.6, 0.4],  # high, cloudy
    (1, 0): [0.5, 0.5],  # low, sunny
    (1, 1): [0.2, 0.8],  # low, cloudy
}

# P(W_t | W_{t-1})
P_W_trans = {
    (0,): [0.7, 0.3],
    (1,): [0.3, 0.7],
}

# P(T_t | T_{t-1}, W_{t-1}, W_t)
# evidence order: T_{t-1}, W_{t-1}, W_t
P_T_trans = {
    (0, 0, 0): [0.8, 0.2],
    (0, 0, 1): [0.5, 0.5],
    (0, 1, 0): [0.6, 0.4],
    (0, 1, 1): [0.4, 0.6],
    (1, 0, 0): [0.5, 0.5],
    (1, 0, 1): [0.3, 0.7],
    (1, 1, 0): [0.3, 0.7],
    (1, 1, 1): [0.2, 0.8],
}

# P(E_t | E_{t-1}, W_t, T_t)
# evidence order: E_{t-1}, W_t, T_t
P_E_trans = {
    (0, 0, 0): [0.8, 0.2],
    (0, 0, 1): [0.7, 0.3],
    (0, 1, 0): [0.6, 0.4],
    (0, 1, 1): [0.4, 0.6],
    (1, 0, 0): [0.6, 0.4],
    (1, 0, 1): [0.4, 0.6],
    (1, 1, 0): [0.3, 0.7],
    (1, 1, 1): [0.2, 0.8],
}


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample(probs) -> int:
    return int(np.random.random() < probs[1])   # 0 or 1

def logp(table, key, value) -> float:
    probs = table[key]
    p = probs[value]
    return np.log(max(p, 1e-300))


# ─────────────────────────────────────────────────────────────────────────────
# Prior sample  (t = 0)
# ─────────────────────────────────────────────────────────────────────────────

def sample_prior() -> Particle:
    W = sample(P_W0[()])
    T = sample(P_T0[(W,)])
    E = sample(P_E0[(W, T)])
    U = sample(P_U[(W,)])
    A = sample(P_A[(E, W)])
    return {'W': W, 'T': T, 'E': E, 'U': U, 'A': A}


# ─────────────────────────────────────────────────────────────────────────────
# Transition  (t -> t+1), with optional interventions
# interventions: dict varname -> value, severing incoming edges for that var
# ─────────────────────────────────────────────────────────────────────────────

def propagate(prev: Particle, interventions: dict) -> Particle:
    cur = {}

    # W_t
    if 'W' in interventions:
        cur['W'] = interventions['W']          # do(W_t) — ignore W_{t-1}
    else:
        cur['W'] = sample(P_W_trans[(prev['W'],)])

    # T_t — parents: T_{t-1}, W_{t-1}, W_t
    if 'T' in interventions:
        cur['T'] = interventions['T']          # do(T_t) — ignore all parents
    else:
        cur['T'] = sample(P_T_trans[(prev['T'], prev['W'], cur['W'])])

    # E_t — parents: E_{t-1}, W_t, T_t
    if 'E' in interventions:
        cur['E'] = interventions['E']          # do(E_t)
    else:
        cur['E'] = sample(P_E_trans[(prev['E'], cur['W'], cur['T'])])

    # Sensors — never intervened in our examples, always sampled
    cur['U'] = sample(P_U[(cur['W'],)])
    cur['A'] = sample(P_A[(cur['E'], cur['W'])])

    return cur


# ─────────────────────────────────────────────────────────────────────────────
# Weighting — log-likelihood of observed variables given current particle
# ─────────────────────────────────────────────────────────────────────────────

def log_weight(particle: Particle, obs: dict) -> float:
    """
    obs keys are full variable names like 'U' or 'A', values are 0/1.
    When do(W), do(T), do(E) are active at this step, the sensors that
    depend on them are still informative (they are downstream, not intervened).
    """
    lw = 0.0
    if 'U' in obs:
        lw += logp(P_U, (particle['W'],), obs['U'])
    if 'A' in obs:
        lw += logp(P_A, (particle['E'], particle['W']), obs['A'])
    return lw


# ─────────────────────────────────────────────────────────────────────────────
# Systematic resampling
# ─────────────────────────────────────────────────────────────────────────────

def systematic_resample(particles, log_weights):
    N = len(particles)
    lw = np.array(log_weights)
    lw -= lw.max()                      # numerical stability
    w  = np.exp(lw)
    w /= w.sum()

    positions = (np.arange(N) + np.random.random()) / N
    cumsum = np.cumsum(w)
    new_particles, i, j = [], 0, 0
    while i < N:
        if positions[i] < cumsum[j]:
            new_particles.append(dict(particles[j]))
            i += 1
        else:
            j += 1
    return new_particles


# ─────────────────────────────────────────────────────────────────────────────
# Main particle filter
#
# obs_schedule    : {t: {'U': val, 'A': val}}   — partial, any subset
# interventions   : {t: {'W': val, ...}}         — do(·) at specific time steps
# query_var       : variable name to query at query_t
# query_t         : time step of interest
# ─────────────────────────────────────────────────────────────────────────────

def particle_filter(obs_schedule: dict, interventions: dict,
                    query_var: str, query_t: int,
                    N: int = 20_000) -> np.ndarray:
    """
    Returns marginal distribution [P(query_var=0), P(query_var=1)]
    computed by particle filtering.
    """
    particles = [sample_prior() for _ in range(N)]

    # Weight/resample at t=0 if there's evidence
    if 0 in obs_schedule:
        lws = [log_weight(p, obs_schedule[0]) for p in particles]
        particles = systematic_resample(particles, lws)

    for t in range(1, query_t + 1):
        iv = interventions.get(t, {})
        particles = [propagate(p, iv) for p in particles]

        if t in obs_schedule:
            lws = [log_weight(p, obs_schedule[t]) for p in particles]
            particles = systematic_resample(particles, lws)

    counts = np.zeros(2)
    for p in particles:
        counts[p[query_var]] += 1
    return counts / counts.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Three examples mirroring exact_inference.py
# ─────────────────────────────────────────────────────────────────────────────

def pf_compare_interventions(N=50_000):
    """
    Mirror of compare_interventions_example():
      1. P(A_5 | U_5=yes)          — observational
      2. P(A_5 | do(W_5=cloudy))   — interventional
    """
    print("\n" + "=" * 80)
    print("PARTICLE FILTER: Observation vs Intervention")
    print("=" * 80)

    # 1. Observational: P(A_5 | U_5=yes)
    print("\n--- Observational Query: P(A_5 | U_5=yes) ---")
    t0 = time.time()
    dist = particle_filter(
        obs_schedule   = {5: {'U': 1}},
        interventions  = {},
        query_var      = 'A',
        query_t        = 5,
        N              = N,
    )
    elapsed = time.time() - t0
    print(f"[PF]  P(A_5=outdoor) = {dist[0]:.6f}")
    print(f"[PF]  P(A_5=indoor)  = {dist[1]:.6f}")
    print(f"[PF]  completed in {elapsed:.4f}s  ({N:,} particles)")

    # 2. Interventional: P(A_5 | do(W_5=cloudy))
    print("\n--- Interventional Query: P(A_5 | do(W_5=cloudy)) ---")
    t0 = time.time()
    dist = particle_filter(
        obs_schedule   = {},
        interventions  = {5: {'W': 1}},   # do(W_5=cloudy)
        query_var      = 'A',
        query_t        = 5,
        N              = N,
    )
    elapsed = time.time() - t0
    print(f"[PF]  P(A_5=outdoor) = {dist[0]:.6f}")
    print(f"[PF]  P(A_5=indoor)  = {dist[1]:.6f}")
    print(f"[PF]  completed in {elapsed:.4f}s  ({N:,} particles)")

    return dist


def pf_problem_query(N=50_000):
    """
    Mirror of problem_query_example():
      P(A_10 | do(W_5=cloudy, T_5=cold, E_5=high), U_{5,6,7}=yes,yes,no, A_{5,6,7}=in,in,out)
    """
    print("\n" + "=" * 80)
    print("PARTICLE FILTER: P(A_10 | do(W_5=cloudy,T_5=cold,E_5=high), U_{5,6,7}, A_{5,6,7})")
    print("=" * 80)

    t0 = time.time()
    dist = particle_filter(
        obs_schedule = {
            5: {'U': 1, 'A': 1},   # umbrella=yes, activity=indoor
            6: {'U': 1, 'A': 1},
            7: {'U': 0, 'A': 0},   # umbrella=no,  activity=outdoor
        },
        interventions = {
            5: {'W': 1, 'T': 1, 'E': 0},  # do(W=cloudy, T=cold, E=high)
        },
        query_var = 'A',
        query_t   = 10,
        N         = N,
    )
    elapsed = time.time() - t0
    print(f"[PF]  P(A_10=outdoor) = {dist[0]:.6f}")
    print(f"[PF]  P(A_10=indoor)  = {dist[1]:.6f}")
    print(f"[PF]  completed in {elapsed:.4f}s  ({N:,} particles)")
    return dist


def pf_marginal_over_time(N=20_000):
    """
    Mirror of marginal_over_time(): P(W_t) for t in 0..10, no evidence.
    Run a single long filter pass and read off marginals at each step.
    """
    print("\n" + "=" * 80)
    print("PARTICLE FILTER: Marginal P(W_t) over time")
    print("=" * 80)

    T_max = 10
    particles = [sample_prior() for _ in range(N)]

    print(f"\n{'Time':<6} {'P(sunny)':<12} {'P(cloudy)':<12}")
    print("-" * 35)

    # t=0
    counts = np.bincount([p['W'] for p in particles], minlength=2)
    dist   = counts / counts.sum()
    print(f"{0:<6} {dist[0]:<12.6f} {dist[1]:<12.6f}")

    for t in range(1, T_max + 1):
        particles = [propagate(p, {}) for p in particles]
        counts = np.bincount([p['W'] for p in particles], minlength=2)
        dist   = counts / counts.sum()
        print(f"{t:<6} {dist[0]:<12.6f} {dist[1]:<12.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def run_exact(suppress_output=False):
    """Import and run exact inference, capturing results."""
    import io, sys
    from exact_inference import build_binary_dbn, exact_inference_query

    results = {}

    # Query 1a: observational P(A_5 | U_5=yes)
    m = build_binary_dbn(T_max=6)
    r, _ = exact_inference_query(m, ['A_5'], evidence={'U_5': 1})
    results['obs'] = r.values.copy()

    # Query 1b: interventional P(A_5 | do(W_5=1))
    m = build_binary_dbn(T_max=6)
    r, _ = exact_inference_query(m, ['A_5'], interventions={'W_5': 1})
    results['int'] = r.values.copy()

    # Query 2: problem query P(A_10 | do(...), evidence)
    m = build_binary_dbn(T_max=11)
    r, _ = exact_inference_query(
        m, ['A_10'],
        evidence={'U_5': 1, 'U_6': 1, 'U_7': 0,
                  'A_5': 1, 'A_6': 1, 'A_7': 0},
        interventions={'W_5': 1, 'T_5': 1, 'E_5': 0},
    )
    results['prob'] = r.values.copy()

    return results


def side_by_side_comparison(N=50_000):
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*22 + "EXACT  vs  PARTICLE FILTER  COMPARISON" + " "*18 + "║")
    print("╚" + "═"*78 + "╝")

    print("\nRunning exact inference (pgmpy)...")
    try:
        exact = run_exact()
        has_exact = True
    except ImportError:
        print("  [!] exact_inference.py / pgmpy not found — skipping exact column")
        has_exact = False

    print(f"\nRunning particle filter (N={N:,})...")

    pf_obs = particle_filter({5: {'U': 1}}, {}, 'A', 5, N)
    pf_int = particle_filter({}, {5: {'W': 1}}, 'A', 5, N)
    pf_prb = particle_filter(
        obs_schedule  = {5: {'U':1,'A':1}, 6: {'U':1,'A':1}, 7: {'U':0,'A':0}},
        interventions = {5: {'W':1,'T':1,'E':0}},
        query_var='A', query_t=10, N=N,
    )

    header = f"  {'Query':<45} {'Value':<10} {'Exact':>10} {'PF':>10} {'|Err|':>8}"
    print("\n" + header)
    print("  " + "─" * (len(header) - 2))

    def row(label, val_label, exact_p, pf_p):
        err = f"{abs(exact_p - pf_p):.6f}" if has_exact else "   n/a"
        ex  = f"{exact_p:.6f}"             if has_exact else "   n/a"
        print(f"  {label:<45} {val_label:<10} {ex:>10} {pf_p:>10.6f} {err:>8}")

    row("P(A_5 | U_5=yes)",
        "outdoor", exact['obs'][0] if has_exact else 0, pf_obs[0])
    row("",
        "indoor",  exact['obs'][1] if has_exact else 0, pf_obs[1])

    row("P(A_5 | do(W_5=cloudy))",
        "outdoor", exact['int'][0] if has_exact else 0, pf_int[0])
    row("",
        "indoor",  exact['int'][1] if has_exact else 0, pf_int[1])

    row("P(A_10 | do(W5=cld,T5=cld,E5=hi), U/A 5-7)",
        "outdoor", exact['prob'][0] if has_exact else 0, pf_prb[0])
    row("",
        "indoor",  exact['prob'][1] if has_exact else 0, pf_prb[1])

    print()


if __name__ == '__main__':
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "PARTICLE FILTER INFERENCE FOR DBN WITH INTERVENTIONS" + " "*8 + "║")
    print("╚" + "="*78 + "╝")

    N = 50_000

    pf_compare_interventions(N)
    pf_problem_query(N)
    pf_marginal_over_time(N)
    side_by_side_comparison(N)

    print("\n" + "="*80)
    print("✓ All particle filter examples completed!")
    print("="*80)
