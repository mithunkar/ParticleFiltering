from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
ALARM_TEMPLATE_PATH = TEMPLATES_DIR / "Alarm.xdsl"


@dataclass(frozen=True)
class StaticNode:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...]
    table: dict[tuple[int, ...], np.ndarray]

    @property
    def cardinality(self) -> int:
        return len(self.states)


@dataclass(frozen=True)
class StaticBN:
    name: str
    nodes: dict[str, StaticNode]
    node_order: tuple[str, ...]

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [
            (parent, node_name)
            for node_name in self.node_order
            for parent in self.nodes[node_name].parents
        ]


@dataclass(frozen=True)
class CPD:
    variable: str
    states: tuple[str, ...]
    parents: tuple[str, ...]
    table: dict[tuple[int, ...], np.ndarray]

    @property
    def cardinality(self) -> int:
        return len(self.states)


@dataclass(frozen=True)
class AlarmDBN:
    state_names: tuple[str, ...]
    states_by_var: dict[str, tuple[str, ...]]
    prior_cpds: dict[str, CPD]
    transition_cpds: dict[str, CPD]
    intra_slice_edges: tuple[tuple[str, str], ...]
    temporal_edges: tuple[tuple[str, str], ...]
    persistence: float


@dataclass(frozen=True)
class UnrolledDBN:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    cpds: dict[str, CPD]
    states_by_var: dict[str, tuple[str, ...]]
    horizon: int


def _assignment_space(cardinalities: list[int]) -> list[tuple[int, ...]]:
    if not cardinalities:
        return [()]
    return list(product(*(range(card) for card in cardinalities)))


def _normalize(dist: np.ndarray) -> np.ndarray:
    total = dist.sum()
    if total <= 0:
        raise ValueError("Encountered a non-positive probability total while normalizing a CPD.")
    return dist / total


def _parse_probabilities(node_elem: ET.Element, parent_cards: list[int], var_card: int) -> dict[tuple[int, ...], np.ndarray]:
    text = node_elem.findtext("probabilities")
    if not text:
        raise ValueError(f"Node {node_elem.attrib['id']} is missing probabilities.")

    values = np.fromiter((float(x) for x in text.split()), dtype=float)
    assignments = _assignment_space(parent_cards)
    expected = len(assignments) * var_card
    if len(values) != expected:
        raise ValueError(
            f"Node {node_elem.attrib['id']} has {len(values)} probabilities, expected {expected}."
        )

    table: dict[tuple[int, ...], np.ndarray] = {}
    for index, assignment in enumerate(assignments):
        start = index * var_card
        table[assignment] = _normalize(values[start : start + var_card].astype(float))
    return table


def load_alarm_bn(path: Path | str = ALARM_TEMPLATE_PATH) -> StaticBN:
    path = Path(path)
    root = ET.parse(path).getroot()
    node_elements = [elem for elem in root.find("nodes") if elem.tag == "cpt"]

    node_order = tuple(elem.attrib["id"] for elem in node_elements)
    state_lookup = {
        elem.attrib["id"]: tuple(state.attrib["id"] for state in elem.findall("state"))
        for elem in node_elements
    }

    nodes: dict[str, StaticNode] = {}
    for elem in node_elements:
        name = elem.attrib["id"]
        parents = tuple((elem.findtext("parents") or "").split())
        states = state_lookup[name]
        parent_cards = [len(state_lookup[parent]) for parent in parents]
        table = _parse_probabilities(elem, parent_cards, len(states))
        nodes[name] = StaticNode(name=name, states=states, parents=parents, table=table)

    return StaticBN(name=path.stem, nodes=nodes, node_order=node_order)


def _sticky_transition_table(node: StaticNode, persistence: float) -> dict[tuple[int, ...], np.ndarray]:
    transition_table: dict[tuple[int, ...], np.ndarray] = {}
    for parent_assignment, baseline in node.table.items():
        for prev_value in range(node.cardinality):
            sticky = np.zeros(node.cardinality, dtype=float)
            sticky[prev_value] = 1.0
            blended = persistence * sticky + (1.0 - persistence) * baseline
            transition_table[(prev_value, *parent_assignment)] = _normalize(blended)
    return transition_table


def build_alarm_dbn(static_bn: StaticBN, persistence: float = 0.8) -> AlarmDBN:
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("Persistence must be between 0.0 and 1.0.")

    prior_cpds: dict[str, CPD] = {}
    transition_cpds: dict[str, CPD] = {}
    states_by_var = {name: node.states for name, node in static_bn.nodes.items()}

    for node_name in static_bn.node_order:
        node = static_bn.nodes[node_name]
        prior_cpds[node_name] = CPD(
            variable=node_name,
            states=node.states,
            parents=node.parents,
            table=node.table,
        )
        transition_cpds[node_name] = CPD(
            variable=node_name,
            states=node.states,
            parents=(f"prev_{node_name}", *node.parents),
            table=_sticky_transition_table(node, persistence),
        )

    intra_edges = tuple(static_bn.edges)
    temporal_edges = tuple((name, name) for name in static_bn.node_order)
    return AlarmDBN(
        state_names=static_bn.node_order,
        states_by_var=states_by_var,
        prior_cpds=prior_cpds,
        transition_cpds=transition_cpds,
        intra_slice_edges=intra_edges,
        temporal_edges=temporal_edges,
        persistence=persistence,
    )


def unroll_alarm_dbn(dbn: AlarmDBN, horizon: int) -> UnrolledDBN:
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")

    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    cpds: dict[str, CPD] = {}
    states_by_var: dict[str, tuple[str, ...]] = {}

    for t in range(horizon):
        for var_name in dbn.state_names:
            timed_var = f"{var_name}_{t}"
            nodes.append(timed_var)
            states_by_var[timed_var] = dbn.states_by_var[var_name]

        for parent, child in dbn.intra_slice_edges:
            edges.append((f"{parent}_{t}", f"{child}_{t}"))

        if t == 0:
            for var_name, cpd in dbn.prior_cpds.items():
                cpds[f"{var_name}_{t}"] = CPD(
                    variable=f"{var_name}_{t}",
                    states=cpd.states,
                    parents=tuple(f"{parent}_{t}" for parent in cpd.parents),
                    table={assignment: dist.copy() for assignment, dist in cpd.table.items()},
                )
        else:
            for parent, child in dbn.temporal_edges:
                edges.append((f"{parent}_{t-1}", f"{child}_{t}"))

            for var_name, cpd in dbn.transition_cpds.items():
                timed_parents = []
                for parent in cpd.parents:
                    if parent.startswith("prev_"):
                        timed_parents.append(f"{parent[5:]}_{t-1}")
                    else:
                        timed_parents.append(f"{parent}_{t}")
                cpds[f"{var_name}_{t}"] = CPD(
                    variable=f"{var_name}_{t}",
                    states=cpd.states,
                    parents=tuple(timed_parents),
                    table={assignment: dist.copy() for assignment, dist in cpd.table.items()},
                )

    return UnrolledDBN(
        nodes=tuple(nodes),
        edges=tuple(edges),
        cpds=cpds,
        states_by_var=states_by_var,
        horizon=horizon,
    )


def summarize_unrolled_dbn(unrolled: UnrolledDBN) -> str:
    cpd_count = len(unrolled.cpds)
    avg_cardinality = np.mean([len(states) for states in unrolled.states_by_var.values()])
    return "\n".join(
        [
            f"Alarm DBN summary (T={unrolled.horizon})",
            f"  Nodes: {len(unrolled.nodes)}",
            f"  Edges: {len(unrolled.edges)}",
            f"  CPDs: {cpd_count}",
            f"  Average cardinality: {avg_cardinality:.2f}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Alarm.xdsl into a full-persistence DBN.")
    parser.add_argument("--horizon", type=int, default=3, help="Number of time slices to unroll.")
    parser.add_argument(
        "--persistence",
        type=float,
        default=0.8,
        help="Sticky prior weight used in transition CPDs.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ALARM_TEMPLATE_PATH,
        help="Path to the Alarm.xdsl template.",
    )
    args = parser.parse_args()

    static_bn = load_alarm_bn(args.template)
    dbn = build_alarm_dbn(static_bn, persistence=args.persistence)
    unrolled = unroll_alarm_dbn(dbn, horizon=args.horizon)

    print(f"Loaded template: {args.template}")
    print(f"Variables: {len(static_bn.node_order)}")
    print(f"Original edges: {len(static_bn.edges)}")
    print(f"Persistence: {dbn.persistence:.2f}")
    print(summarize_unrolled_dbn(unrolled))


if __name__ == "__main__":
    main()
