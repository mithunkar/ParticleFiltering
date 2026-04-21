from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from alarm_dbn import build_alarm_dbn, load_alarm_bn, unroll_alarm_dbn


@dataclass(frozen=True)
class TimedIntervention:
    time: int
    variable: str
    value: str


def _state_index(states_by_var: dict[str, tuple[str, ...]], var: str, value: str | int) -> int:
    if isinstance(value, int):
        return value
    states = states_by_var[var]
    if value not in states:
        raise ValueError(f"Invalid value '{value}' for {var}. Valid states: {states}")
    return states.index(value)


def _to_pgmpy(unrolled):
    model = DiscreteBayesianNetwork(list(unrolled.edges))
    cpds = []
    for var in unrolled.nodes:
        cpd = unrolled.cpds[var]
        parents = list(cpd.parents)
        parent_cards = [len(unrolled.states_by_var[p]) for p in parents]
        assignments = list(product(*(range(card) for card in parent_cards))) if parents else [()]
        values = [[] for _ in range(cpd.cardinality)]
        for assignment in assignments:
            dist = cpd.table[assignment]
            for state_idx in range(cpd.cardinality):
                values[state_idx].append(float(dist[state_idx]))

        if parents:
            cpds.append(
                TabularCPD(
                    variable=var,
                    variable_card=cpd.cardinality,
                    values=values,
                    evidence=parents,
                    evidence_card=parent_cards,
                )
            )
        else:
            cpds.append(
                TabularCPD(
                    variable=var,
                    variable_card=cpd.cardinality,
                    values=values,
                )
            )

    model.add_cpds(*cpds)
    if not model.check_model():
        raise ValueError("Constructed pgmpy model is invalid.")
    return model


def _apply_interventions(model: DiscreteBayesianNetwork, unrolled, interventions: list[TimedIntervention]):
    if not interventions:
        return model

    new_model = DiscreteBayesianNetwork()
    new_model.add_nodes_from(model.nodes())
    new_model.add_edges_from(model.edges())
    for cpd in model.get_cpds():
        new_model.add_cpds(cpd)

    for intervention in interventions:
        timed_var = f"{intervention.variable}_{intervention.time}"
        states = unrolled.states_by_var[timed_var]
        forced_idx = states.index(intervention.value)

        for parent in list(new_model.get_parents(timed_var)):
            new_model.remove_edge(parent, timed_var)

        existing = new_model.get_cpds(timed_var)
        if existing is not None:
            new_model.remove_cpds(existing)

        values = [[0.0] for _ in range(len(states))]
        values[forced_idx][0] = 1.0
        new_model.add_cpds(
            TabularCPD(variable=timed_var, variable_card=len(states), values=values)
        )

    if not new_model.check_model():
        raise ValueError("Intervened pgmpy model is invalid.")
    return new_model


def exact_query(unrolled, query_var: str, query_t: int,
                evidence: dict[int, dict[str, str]] | None = None,
                interventions: list[TimedIntervention] | None = None):
    evidence = evidence or {}
    interventions = interventions or []

    model = _to_pgmpy(unrolled)
    model = _apply_interventions(model, unrolled, interventions)

    timed_query = f"{query_var}_{query_t}"
    pgm_evidence = {}
    for t, assignments in evidence.items():
        for var, value in assignments.items():
            timed_var = f"{var}_{t}"
            pgm_evidence[timed_var] = _state_index(unrolled.states_by_var, timed_var, value)

    t0 = time.time()
    result = VariableElimination(model).query(
        variables=[timed_query],
        evidence=pgm_evidence,
        show_progress=False,
    )
    return result.values.copy(), time.time() - t0


def _systematic_resample(particles, log_weights):
    lw = np.array(log_weights, dtype=float)
    lw -= lw.max()
    w = np.exp(lw)
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w /= w_sum

    n = len(particles)
    positions = (np.arange(n) + np.random.random()) / n
    cumsum = np.cumsum(w)
    new_particles = []
    i = j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            new_particles.append(dict(particles[j]))
            i += 1
        else:
            j += 1
    return new_particles


def _topological_state_order(dbn):
    graph = nx.DiGraph()
    graph.add_nodes_from(dbn.state_names)
    graph.add_edges_from(dbn.intra_slice_edges)
    return tuple(nx.topological_sort(graph))


def particle_query(dbn, query_var: str, query_t: int,
                   evidence: dict[int, dict[str, str]] | None = None,
                   interventions: list[TimedIntervention] | None = None,
                   n_particles: int = 30_000,
                   seed: int = 0):
    evidence = evidence or {}
    interventions = interventions or []
    rng = np.random.default_rng(seed)

    state_order = _topological_state_order(dbn)
    intervention_map: dict[int, dict[str, int]] = {}
    for itv in interventions:
        intervention_map.setdefault(itv.time, {})[itv.variable] = _state_index(
            dbn.states_by_var, itv.variable, itv.value
        )

    def draw(dist: np.ndarray) -> int:
        return int(rng.choice(len(dist), p=dist))

    def sample_prior():
        p = {}
        for var in state_order:
            if 0 in intervention_map and var in intervention_map[0]:
                p[var] = intervention_map[0][var]
                continue
            cpd = dbn.prior_cpds[var]
            parent_vals = tuple(p[parent] for parent in cpd.parents)
            p[var] = draw(cpd.table[parent_vals])
        return p

    def propagate(prev, t):
        cur = {}
        iv = intervention_map.get(t, {})
        for var in state_order:
            if var in iv:
                cur[var] = iv[var]
                continue
            cpd = dbn.transition_cpds[var]
            parent_assignment = []
            for parent in cpd.parents:
                if parent.startswith("prev_"):
                    parent_assignment.append(prev[parent[5:]])
                else:
                    parent_assignment.append(cur[parent])
            cur[var] = draw(cpd.table[tuple(parent_assignment)])
        return cur

    def log_weight(state, t):
        assignments = evidence.get(t, {})
        if not assignments:
            return 0.0
        lw = 0.0
        for var, value in assignments.items():
            target = _state_index(dbn.states_by_var, var, value)
            lw += 0.0 if state[var] == target else -1e9
        return lw

    t0 = time.time()
    particles = [sample_prior() for _ in range(n_particles)]

    lws = [log_weight(p, 0) for p in particles]
    if any(w < 0 for w in lws):
        particles = _systematic_resample(particles, lws)

    for t in range(1, query_t + 1):
        particles = [propagate(p, t) for p in particles]
        lws = [log_weight(p, t) for p in particles]
        if any(w < 0 for w in lws):
            particles = _systematic_resample(particles, lws)

    card = len(dbn.states_by_var[query_var])
    counts = np.zeros(card, dtype=float)
    for p in particles:
        counts[p[query_var]] += 1.0
    dist = counts / counts.sum()
    return dist, time.time() - t0


def _format_dist(states, dist):
    parts = [f"{state}={prob:.6f}" for state, prob in zip(states, dist)]
    return ", ".join(parts)


def run_default_experiments(horizon: int, n_particles: int, persistence: float):
    static_bn = load_alarm_bn()
    dbn = build_alarm_dbn(static_bn, persistence=persistence)
    unrolled = unroll_alarm_dbn(dbn, horizon=horizon)

    experiments = [
        {
            "name": "Observational P(HR_2 | BP_2=Low)",
            "query_var": "HR",
            "query_t": 2,
            "evidence": {2: {"BP": "Low"}},
            "interventions": [],
        },
        {
            "name": "Interventional P(HR_2 | do(Anaphylaxis_1=True))",
            "query_var": "HR",
            "query_t": 2,
            "evidence": {},
            "interventions": [TimedIntervention(1, "Anaphylaxis", "True")],
        },
        {
            "name": "Mixed P(BP_2 | do(Disconnect_1=True), VentTube_2=Zero)",
            "query_var": "BP",
            "query_t": 2,
            "evidence": {2: {"VentTube": "Zero"}},
            "interventions": [TimedIntervention(1, "Disconnect", "True")],
        },
    ]

    print("=" * 96)
    print("ALARM DBN: Exact vs Particle Filtering")
    print("=" * 96)
    print(f"Horizon: {horizon}, Persistence: {persistence:.2f}, Particles: {n_particles:,}")
    print()

    for exp in experiments:
        print("-" * 96)
        print(exp["name"])
        exact_dist, exact_time = exact_query(
            unrolled,
            query_var=exp["query_var"],
            query_t=exp["query_t"],
            evidence=exp["evidence"],
            interventions=exp["interventions"],
        )
        pf_dist, pf_time = particle_query(
            dbn,
            query_var=exp["query_var"],
            query_t=exp["query_t"],
            evidence=exp["evidence"],
            interventions=exp["interventions"],
            n_particles=n_particles,
            seed=0,
        )

        states = dbn.states_by_var[exp["query_var"]]
        l1 = float(np.abs(exact_dist - pf_dist).sum())
        print(f"Exact ({exact_time:.4f}s):    {_format_dist(states, exact_dist)}")
        print(f"Particle ({pf_time:.4f}s): {_format_dist(states, pf_dist)}")
        print(f"L1 distance: {l1:.6f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run exact vs particle experiments on Alarm DBN")
    parser.add_argument("--horizon", type=int, default=3, help="Unrolled horizon (>= query time + 1)")
    parser.add_argument("--particles", type=int, default=30_000, help="Number of particles")
    parser.add_argument("--persistence", type=float, default=0.8, help="Transition persistence")
    args = parser.parse_args()

    run_default_experiments(
        horizon=args.horizon,
        n_particles=args.particles,
        persistence=args.persistence,
    )


if __name__ == "__main__":
    main()