"""
Dynamic Bayesian Network (DBN) — Discrete Variables Only
with Particle Filtering + Future Prediction + Full DBN Visualization

AIMA (Russell & Norvig) style.

Example DBN: Simple weather/activity model
Variables per time slice (all discrete):
  W  - Weather:   {sunny, cloudy, rainy}
  T  - Temp:      {hot, mild, cold}
  U  - Umbrella:  {yes, no}          ← observed
  A  - Activity:  {outdoor, indoor}  ← observed
  E  - Energy:    {high, low}        (latent, affects activity)

Dependencies:
  W_t | W_{t-1}
  T_t | T_{t-1}, W_t
  E_t | E_{t-1}, W_t, T_t
  U_t | W_t                   (sensor)
  A_t | E_t, W_t              (sensor)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
from dataclasses import dataclass
from typing import Callable, Any
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Core DBN Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Variable:
    name: str
    values: list   # discrete domain

@dataclass
class CPD:
    variable: Variable
    parents: list[str]        # parent var names; prefix 'prev_' = prior slice
    sample_fn:  Callable      # (parent_vals: dict) -> value
    logprob_fn: Callable      # (value, parent_vals: dict) -> log prob

@dataclass
class DBN:
    state_vars:       list[Variable]
    obs_vars:         list[Variable]
    prior_cpds:       dict[str, CPD]
    transition_cpds:  dict[str, CPD]
    sensor_cpds:      dict[str, CPD]

Particle = dict   # varname -> value


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_discrete_cpd(var: Variable, parents: list[str],
                      table: dict) -> CPD:
    """
    table maps tuple(parent_values) -> {value: prob}
    For root nodes, table maps () -> {value: prob}.
    """
    def sample_fn(p):
        key = tuple(p[pn] for pn in parents)
        dist = table[key]
        vals, probs = zip(*dist.items())
        return np.random.choice(vals, p=probs)

    def logprob_fn(v, p):
        key = tuple(p[pn] for pn in parents)
        dist = table[key]
        return np.log(dist.get(v, 1e-12))

    return CPD(var, parents, sample_fn, logprob_fn)


# ─────────────────────────────────────────────────────────────
# Build the Weather DBN
# ─────────────────────────────────────────────────────────────

def build_weather_dbn() -> DBN:
    W = Variable('W', ['sunny', 'cloudy', 'rainy'])
    T = Variable('T', ['hot', 'mild', 'cold'])
    E = Variable('E', ['high', 'low'])
    U = Variable('U', ['yes', 'no'])
    A = Variable('A', ['outdoor', 'indoor'])

    # ── Priors ──────────────────────────────────────────────

    prior_W = make_discrete_cpd(W, [], {
        (): {'sunny': 0.5, 'cloudy': 0.3, 'rainy': 0.2}
    })

    prior_T = make_discrete_cpd(T, [], {
        (): {'hot': 0.3, 'mild': 0.5, 'cold': 0.2}
    })

    prior_E = make_discrete_cpd(E, [], {
        (): {'high': 0.6, 'low': 0.4}
    })

    # ── Transitions ─────────────────────────────────────────

    # W_t | W_{t-1}  (tendency to persist)
    trans_W = make_discrete_cpd(W, ['prev_W'], {
        ('sunny',): {'sunny': 0.7, 'cloudy': 0.2, 'rainy': 0.1},
        ('cloudy',): {'sunny': 0.3, 'cloudy': 0.4, 'rainy': 0.3},
        ('rainy',): {'sunny': 0.1, 'cloudy': 0.3, 'rainy': 0.6},
    })

    # T_t | T_{t-1}, W_t
    trans_T = make_discrete_cpd(T, ['prev_T', 'W'], {
        ('hot',   'sunny'): {'hot': 0.7, 'mild': 0.2, 'cold': 0.1},
        ('hot',   'cloudy'):{'hot': 0.4, 'mild': 0.4, 'cold': 0.2},
        ('hot',   'rainy'): {'hot': 0.2, 'mild': 0.5, 'cold': 0.3},
        ('mild',  'sunny'): {'hot': 0.3, 'mild': 0.5, 'cold': 0.2},
        ('mild',  'cloudy'):{'hot': 0.2, 'mild': 0.5, 'cold': 0.3},
        ('mild',  'rainy'): {'hot': 0.1, 'mild': 0.4, 'cold': 0.5},
        ('cold',  'sunny'): {'hot': 0.1, 'mild': 0.4, 'cold': 0.5},
        ('cold',  'cloudy'):{'hot': 0.1, 'mild': 0.3, 'cold': 0.6},
        ('cold',  'rainy'): {'hot': 0.0, 'mild': 0.2, 'cold': 0.8},
    })

    # E_t | E_{t-1}, W_t, T_t
    trans_E = make_discrete_cpd(E, ['prev_E', 'W', 'T'], {
        ('high', 'sunny',  'hot'):  {'high': 0.8, 'low': 0.2},
        ('high', 'sunny',  'mild'): {'high': 0.8, 'low': 0.2},
        ('high', 'sunny',  'cold'): {'high': 0.6, 'low': 0.4},
        ('high', 'cloudy', 'hot'):  {'high': 0.6, 'low': 0.4},
        ('high', 'cloudy', 'mild'): {'high': 0.6, 'low': 0.4},
        ('high', 'cloudy', 'cold'): {'high': 0.4, 'low': 0.6},
        ('high', 'rainy',  'hot'):  {'high': 0.4, 'low': 0.6},
        ('high', 'rainy',  'mild'): {'high': 0.3, 'low': 0.7},
        ('high', 'rainy',  'cold'): {'high': 0.2, 'low': 0.8},
        ('low',  'sunny',  'hot'):  {'high': 0.5, 'low': 0.5},
        ('low',  'sunny',  'mild'): {'high': 0.5, 'low': 0.5},
        ('low',  'sunny',  'cold'): {'high': 0.3, 'low': 0.7},
        ('low',  'cloudy', 'hot'):  {'high': 0.3, 'low': 0.7},
        ('low',  'cloudy', 'mild'): {'high': 0.3, 'low': 0.7},
        ('low',  'cloudy', 'cold'): {'high': 0.2, 'low': 0.8},
        ('low',  'rainy',  'hot'):  {'high': 0.2, 'low': 0.8},
        ('low',  'rainy',  'mild'): {'high': 0.1, 'low': 0.9},
        ('low',  'rainy',  'cold'): {'high': 0.05,'low': 0.95},
    })

    # ── Sensors ─────────────────────────────────────────────

    # U_t | W_t
    sensor_U = make_discrete_cpd(U, ['W'], {
        ('sunny',): {'yes': 0.05, 'no': 0.95},
        ('cloudy',):{'yes': 0.40, 'no': 0.60},
        ('rainy',): {'yes': 0.90, 'no': 0.10},
    })

    # A_t | E_t, W_t
    sensor_A = make_discrete_cpd(A, ['E', 'W'], {
        ('high', 'sunny'):  {'outdoor': 0.9, 'indoor': 0.1},
        ('high', 'cloudy'): {'outdoor': 0.6, 'indoor': 0.4},
        ('high', 'rainy'):  {'outdoor': 0.2, 'indoor': 0.8},
        ('low',  'sunny'):  {'outdoor': 0.5, 'indoor': 0.5},
        ('low',  'cloudy'): {'outdoor': 0.2, 'indoor': 0.8},
        ('low',  'rainy'):  {'outdoor': 0.05,'indoor': 0.95},
    })

    return DBN(
        state_vars      = [W, T, E],
        obs_vars        = [U, A],
        prior_cpds      = {'W': prior_W, 'T': prior_T, 'E': prior_E},
        transition_cpds = {'W': trans_W, 'T': trans_T, 'E': trans_E},
        sensor_cpds     = {'U': sensor_U, 'A': sensor_A},
    )


# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────

def simulate_dbn(dbn: DBN, T: int = 10):
    state = {}
    for vname, cpd in dbn.prior_cpds.items():
        pv = {pn: state[pn] for pn in cpd.parents}
        state[vname] = cpd.sample_fn(pv)

    traj, obs_seq = [state.copy()], []

    for _ in range(1, T):
        new_state = {}
        for vname, cpd in dbn.transition_cpds.items():
            pv = {}
            for pn in cpd.parents:
                pv[pn] = state[pn[5:]] if pn.startswith('prev_') else new_state[pn]
            new_state[vname] = cpd.sample_fn(pv)
        state = new_state
        traj.append(state.copy())

    for s in traj:
        o = {}
        for oname, cpd in dbn.sensor_cpds.items():
            pv = {pn: s[pn] for pn in cpd.parents}
            o[oname] = cpd.sample_fn(pv)
        obs_seq.append(o)

    return traj, obs_seq


# ─────────────────────────────────────────────────────────────
# Particle Filtering  +  Free Prediction
# ─────────────────────────────────────────────────────────────

def systematic_resample(particles, weights):
    N = len(particles)
    w = np.array(weights, dtype=float)
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


def propagate(particle: Particle, trans_cpds: dict) -> Particle:
    new_p = {}
    for vname, cpd in trans_cpds.items():
        pv = {}
        for pn in cpd.parents:
            pv[pn] = particle[pn[5:]] if pn.startswith('prev_') else new_p[pn]
        new_p[vname] = cpd.sample_fn(pv)
    return new_p


def particle_filter_with_prediction(dbn: DBN, observations: list[dict],
                                    evidence_until: int, N: int = 1000):
    """
    Run particle filter with reweighting up to `evidence_until` (inclusive),
    then propagate freely (no reweighting) for remaining steps.

    Returns list of particle sets, one per time step.
    """
    def sample_prior() -> Particle:
        p = {}
        for vname, cpd in dbn.prior_cpds.items():
            pv = {pn: p[pn] for pn in cpd.parents}
            p[vname] = cpd.sample_fn(pv)
        return p

    def weight_particle(particle: Particle, obs: dict) -> float:
        log_w = 0.0
        for oname, oval in obs.items():
            cpd = dbn.sensor_cpds[oname]
            pv = {pn: particle[pn] for pn in cpd.parents}
            log_w += cpd.logprob_fn(oval, pv)
        return np.exp(log_w)

    particles = [sample_prior() for _ in range(N)]
    results = []

    for t, obs in enumerate(observations):
        if t > 0:
            particles = [propagate(p, dbn.transition_cpds) for p in particles]

        if t <= evidence_until:
            # Filtered: weight by observation
            weights = [weight_particle(p, obs) for p in particles]
            if sum(weights) == 0:
                weights = [1.0] * N
            particles = systematic_resample(particles, weights)
            mode = 'filtered'
        else:
            # Prediction: let particles run free
            mode = 'predicted'

        results.append((mode, [dict(p) for p in particles]))

    return results


# ─────────────────────────────────────────────────────────────
# Distribution Utilities
# ─────────────────────────────────────────────────────────────

def marginal(particles: list[Particle], var: str) -> dict:
    counts = defaultdict(int)
    for p in particles:
        counts[p[var]] += 1
    total = len(particles)
    return {k: v / total for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────
# DBN Structure Visualization  (two-slice template)
# ─────────────────────────────────────────────────────────────

def visualize_dbn_structure(dbn: DBN, ax):
    """Draw the two-slice DBN template showing all variable connections."""
    G = nx.DiGraph()

    state_names = [v.name for v in dbn.state_vars]
    obs_names   = [v.name for v in dbn.obs_vars]
    all_names   = state_names + obs_names

    # Node positions: two columns (t-1 left, t right)
    # y positions spaced evenly
    n = len(all_names)
    ys = np.linspace(1, 0, n)
    pos = {}
    for i, name in enumerate(all_names):
        pos[f'{name}_prev'] = (0.0, ys[i])
        pos[f'{name}_cur']  = (1.0, ys[i])

    # Add all nodes
    for name in all_names:
        G.add_node(f'{name}_prev')
        G.add_node(f'{name}_cur')

    # Intra-slice edges (current slice): from sensor/transition parents
    intra_edges = []
    cross_edges = []

    for vname, cpd in {**dbn.transition_cpds, **dbn.sensor_cpds}.items():
        for pn in cpd.parents:
            if pn.startswith('prev_'):
                cross_edges.append((f'{pn[5:]}_prev', f'{vname}_cur'))
                G.add_edge(f'{pn[5:]}_prev', f'{vname}_cur')
            else:
                intra_edges.append((f'{pn}_cur', f'{vname}_cur'))
                G.add_edge(f'{pn}_cur', f'{vname}_cur')

    # Color coding
    state_color   = '#4A90D9'
    obs_color     = '#E8A838'
    prev_alpha    = 0.45

    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        name_part = node.replace('_prev', '').replace('_cur', '')
        is_obs    = name_part in obs_names
        is_prev   = node.endswith('_prev')
        base_col  = obs_color if is_obs else state_color
        node_colors.append(base_col)
        node_sizes.append(900 if not is_prev else 700)

    # Draw
    ax.set_facecolor('#1a1a2e')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)

    # Cross-slice edges (dashed, lighter)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax,
                           edge_color='#aaaacc', style='dashed',
                           arrows=True, arrowsize=14,
                           connectionstyle='arc3,rad=0.05',
                           width=1.5, alpha=0.7)

    # Intra-slice edges (solid)
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=ax,
                           edge_color='#ffffff', style='solid',
                           arrows=True, arrowsize=14,
                           connectionstyle='arc3,rad=0.1',
                           width=1.8, alpha=0.85)

    # Labels
    labels = {node: node.replace('_prev', '\n(t-1)').replace('_cur', '\n(t)')
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=7.5, font_color='white', font_weight='bold')

    # Slice labels
    ax.text(0.0, 1.07, 'Slice  t−1', transform=ax.transAxes,
            ha='center', color='#aaaacc', fontsize=10, style='italic')
    ax.text(1.0, 1.07, 'Slice  t', transform=ax.transAxes,
            ha='center', color='#aaaacc', fontsize=10, style='italic')

    # Legend
    legend_elements = [
        mpatches.Patch(color=state_color, label='Latent state'),
        mpatches.Patch(color=obs_color,   label='Observed'),
        plt.Line2D([0],[0], color='#aaaacc', linestyle='--', label='Cross-slice'),
        plt.Line2D([0],[0], color='#ffffff', linestyle='-',  label='Intra-slice'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              facecolor='#2a2a4e', edgecolor='none',
              labelcolor='white', fontsize=8, ncol=2,
              bbox_to_anchor=(0.5, -0.12))

    ax.set_title('DBN Two-Slice Template', color='white',
                 fontsize=12, fontweight='bold', pad=14)
    ax.axis('off')


# ─────────────────────────────────────────────────────────────
# Results Visualization
# ─────────────────────────────────────────────────────────────

COLORS = {
    # Weather
    'sunny': '#F4C430', 'cloudy': '#8899AA', 'rainy': '#4488CC',
    # Temp
    'hot': '#E05A2B', 'mild': '#6DB56D', 'cold': '#5588EE',
    # Energy
    'high': '#AA66CC', 'low': '#997755',
}

def plot_marginals(results, true_traj, obs_seq, evidence_until, ax_W, ax_T, ax_E):
    T = len(results)
    ts = list(range(T))

    for var, ax, title in [('W', ax_W, 'Weather  W'),
                            ('T', ax_T, 'Temperature  T'),
                            ('E', ax_E, 'Energy  E')]:
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

        var_obj = next(v for v in [*dbn.state_vars, *dbn.obs_vars] if v.name == var)
        for val in var_obj.values:
            probs = [marginal(particles, var).get(val, 0.0)
                     for _, particles in results]
            col = COLORS.get(val, '#cccccc')
            ax.plot(ts, probs, color=col, linewidth=2, label=val)
            ax.fill_between(ts, probs, alpha=0.15, color=col)

        # True trajectory dots
        for t, state in enumerate(true_traj):
            ax.scatter(t, 1.02, marker='v', s=40,
                       color=COLORS.get(state[var], '#ffffff'),
                       zorder=5, clip_on=False)

        # Prediction boundary
        ax.axvline(evidence_until + 0.5, color='#ff6666',
                   linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(evidence_until + 0.55, 0.95, 'prediction →',
                color='#ff6666', fontsize=7, va='top',
                transform=ax.get_xaxis_transform())

        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, T - 1)
        ax.set_ylabel('P', color='#aaaacc', fontsize=8)
        ax.set_xlabel('Time step', color='#aaaacc', fontsize=8)
        ax.legend(loc='upper right', facecolor='#2a2a4e',
                  edgecolor='none', labelcolor='white', fontsize=7)
        ax.yaxis.label.set_color('#aaaacc')


def plot_observations(obs_seq, ax):
    ax.set_facecolor('#1a1a2e')
    ax.set_title('Observations  (U = umbrella, A = activity)',
                 color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    T = len(obs_seq)
    U_vals = [obs['U'] for obs in obs_seq]
    A_vals = [obs['A'] for obs in obs_seq]

    u_map = {'yes': 1, 'no': 0}
    a_map = {'outdoor': 1, 'indoor': 0}

    ax.step(range(T), [u_map[v] for v in U_vals],
            where='mid', color='#E8A838', linewidth=2, label='Umbrella (yes=1)')
    ax.step(range(T), [a_map[v] for v in A_vals],
            where='mid', color='#88CCAA', linewidth=2, linestyle='--',
            label='Activity (outdoor=1)')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['no/indoor', 'yes/outdoor'], color='white', fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.set_xlabel('Time step', color='#aaaacc', fontsize=8)
    ax.legend(loc='upper right', facecolor='#2a2a4e',
              edgecolor='none', labelcolor='white', fontsize=7)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(7)
    T              = 10
    EVIDENCE_UNTIL = 6   # filter up to t=6 inclusive, predict t=7..9
    N_PARTICLES    = 2000

    dbn = build_weather_dbn()
    true_traj, obs_seq = simulate_dbn(dbn, T)
    results = particle_filter_with_prediction(
        dbn, obs_seq, evidence_until=EVIDENCE_UNTIL, N=N_PARTICLES)

    # ── Print table ──────────────────────────────────────────
    print(f"\n{'t':>3}  {'true_W':>8} {'est_W':>10}  "
          f"{'true_T':>6} {'est_T':>8}  "
          f"{'true_E':>6} {'est_E':>8}  "
          f"{'obs_U':>6} {'obs_A':>8}  mode")
    print('─' * 95)

    for t, (mode, particles) in enumerate(results):
        mW = marginal(particles, 'W')
        mT = marginal(particles, 'T')
        mE = marginal(particles, 'E')
        best_W = max(mW, key=mW.get)
        best_T = max(mT, key=mT.get)
        best_E = max(mE, key=mE.get)
        marker = '  ←' if mode == 'predicted' else ''
        print(f"{t:>3}  {true_traj[t]['W']:>8} {best_W:>10}  "
              f"{true_traj[t]['T']:>6} {best_T:>8}  "
              f"{true_traj[t]['E']:>6} {best_E:>8}  "
              f"{obs_seq[t]['U']:>6} {obs_seq[t]['A']:>8}  "
              f"{mode}{marker}")

    # ── Plot ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor='#12122a')
    fig.suptitle('Discrete DBN  ·  Particle Filtering + Prediction',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.07,
                           wspace=0.35, hspace=0.55)

    ax_struct = fig.add_subplot(gs[0:2, 0])
    ax_obs    = fig.add_subplot(gs[2, 0])
    ax_W      = fig.add_subplot(gs[0, 1:])
    ax_T      = fig.add_subplot(gs[1, 1:])
    ax_E      = fig.add_subplot(gs[2, 1:])

    visualize_dbn_structure(dbn, ax_struct)

    _, particle_sets = zip(*results)
    plot_marginals(results, true_traj, obs_seq,
                  EVIDENCE_UNTIL, ax_W, ax_T, ax_E)
    plot_observations(obs_seq, ax_obs)

    # Small legend for true-value triangles
    fig.text(0.54, 0.005,
             '▾ triangle = true value at that step   '
             '─ ─  red dashed = prediction boundary',
             color='#aaaacc', fontsize=8, ha='center')

    plt.savefig('dbn_particle_filter.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print('\nPlot saved to dbn_particle_filter.png')
    plt.show()
