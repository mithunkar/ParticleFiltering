from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-particlefiltering"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch


BINARY_STATES = (0, 1)
PROBABILITY_PLACEHOLDER = "?"


@dataclass(frozen=True)
class ADMGConfig:
    node_names: tuple[str, ...]
    topological_order: tuple[str, ...]
    directed_adj: np.ndarray
    bidirected_adj: np.ndarray
    states_by_var: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class ADMGStructure:
    node_names: tuple[str, ...]
    topological_order: tuple[str, ...]
    directed_adj: np.ndarray
    bidirected_adj: np.ndarray
    states_by_var: dict[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        directed = _as_binary_matrix(self.directed_adj, name="directed_adj")
        bidirected = _as_binary_matrix(self.bidirected_adj, name="bidirected_adj")
        object.__setattr__(self, "directed_adj", directed)
        object.__setattr__(self, "bidirected_adj", bidirected)
        validate_admg_structure(self)

    @property
    def index_by_name(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.node_names)}

    @property
    def directed_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (src, dst)
            for src_index, src in enumerate(self.node_names)
            for dst_index, dst in enumerate(self.node_names)
            if self.directed_adj[src_index, dst_index]
        )

    @property
    def bidirected_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (left, right)
            for left_index, left in enumerate(self.node_names)
            for right_index, right in enumerate(self.node_names)
            if left_index < right_index and self.bidirected_adj[left_index, right_index]
        )


@dataclass(frozen=True)
class SkeletonTableRow:
    conditioning_values: tuple[Any, ...]
    scope_values: tuple[Any, ...]
    probability_placeholder: str = PROBABILITY_PLACEHOLDER


@dataclass(frozen=True)
class NodeKernelSkeleton:
    variable: str
    conditioned_on: tuple[str, ...]
    states_by_var: dict[str, tuple[Any, ...]]
    rows: tuple[SkeletonTableRow, ...]
    probability_placeholder: str = PROBABILITY_PLACEHOLDER

    @property
    def scope(self) -> tuple[str, ...]:
        return (self.variable,)

    def conditioning_assignments(self) -> list[tuple[Any, ...]]:
        return assignment_space(self.conditioned_on, self.states_by_var)

    def state_labels(self) -> tuple[Any, ...]:
        return self.states_by_var[self.variable]


@dataclass(frozen=True)
class TianPearlFactorSkeleton:
    name: str
    component: frozenset[str]
    scope: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    term_variables: tuple[str, ...]
    symbolic_factorization: str
    states_by_var: dict[str, tuple[Any, ...]]
    rows: tuple[SkeletonTableRow, ...]
    probability_placeholder: str = PROBABILITY_PLACEHOLDER

    def scope_assignments(self) -> list[tuple[Any, ...]]:
        return assignment_space(self.scope, self.states_by_var)

    def conditioning_assignments(self) -> list[tuple[Any, ...]]:
        return assignment_space(self.conditioned_on, self.states_by_var)


@dataclass(frozen=True)
class InitializedADMGModel:
    graph: ADMGStructure
    node_kernels: dict[str, NodeKernelSkeleton]
    factors: dict[str, TianPearlFactorSkeleton]


def _as_binary_matrix(values: np.ndarray | list[list[int]], *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=int)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if not np.all(np.isin(matrix, (0, 1))):
        raise ValueError(f"{name} must contain only 0/1 entries.")
    return matrix


def assignment_space(
    variables: tuple[str, ...],
    states_by_var: dict[str, tuple[Any, ...]],
) -> list[tuple[Any, ...]]:
    if not variables:
        return [()]
    return list(product(*(states_by_var[var] for var in variables)))


def build_adjacency_matrix(
    node_names: tuple[str, ...],
    edges: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    *,
    symmetric: bool = False,
) -> np.ndarray:
    index_by_name = {name: index for index, name in enumerate(node_names)}
    matrix = np.zeros((len(node_names), len(node_names)), dtype=int)
    for left, right in edges:
        if left not in index_by_name or right not in index_by_name:
            raise ValueError(f"Unknown edge endpoint in ({left}, {right}).")
        left_index = index_by_name[left]
        right_index = index_by_name[right]
        matrix[left_index, right_index] = 1
        if symmetric:
            matrix[right_index, left_index] = 1
    return matrix


def validate_admg_structure(graph: ADMGStructure) -> None:
    node_count = len(graph.node_names)
    if len(set(graph.node_names)) != node_count:
        raise ValueError("node_names must be unique.")
    if set(graph.topological_order) != set(graph.node_names) or len(graph.topological_order) != node_count:
        raise ValueError("topological_order must be a permutation of node_names.")

    for matrix_name, matrix in (
        ("directed_adj", graph.directed_adj),
        ("bidirected_adj", graph.bidirected_adj),
    ):
        if matrix.shape != (node_count, node_count):
            raise ValueError(
                f"{matrix_name} must have shape {(node_count, node_count)}, got {matrix.shape}."
            )

    if np.any(np.diag(graph.directed_adj) != 0):
        raise ValueError("directed_adj must have a zero diagonal.")
    if np.any(np.diag(graph.bidirected_adj) != 0):
        raise ValueError("bidirected_adj must have a zero diagonal.")
    if not np.array_equal(graph.bidirected_adj, graph.bidirected_adj.T):
        raise ValueError("bidirected_adj must be symmetric.")

    topo_position = {name: index for index, name in enumerate(graph.topological_order)}
    for src, dst in graph.directed_edges:
        if topo_position[src] >= topo_position[dst]:
            raise ValueError(
                f"Directed edge {src} -> {dst} violates the supplied topological order."
            )

    if set(graph.states_by_var) != set(graph.node_names):
        raise ValueError("states_by_var must define states for every node.")
    for node_name, states in graph.states_by_var.items():
        if len(states) == 0:
            raise ValueError(f"{node_name} must have at least one state.")


def build_graph_from_config(config: ADMGConfig) -> ADMGStructure:
    return ADMGStructure(
        node_names=config.node_names,
        topological_order=config.topological_order,
        directed_adj=config.directed_adj,
        bidirected_adj=config.bidirected_adj,
        states_by_var=config.states_by_var,
    )


def directed_parents(graph: ADMGStructure, node_name: str) -> tuple[str, ...]:
    node_index = graph.index_by_name[node_name]
    return tuple(
        other
        for other in graph.topological_order
        if graph.directed_adj[graph.index_by_name[other], node_index]
    )


def bidirected_neighbors(graph: ADMGStructure, node_name: str) -> tuple[str, ...]:
    node_index = graph.index_by_name[node_name]
    return tuple(
        other
        for other in graph.topological_order
        if other != node_name and graph.bidirected_adj[node_index, graph.index_by_name[other]]
    )

# For each node V_i:

# take the prefix graph up to V_i
# find the c-component containing V_i
# start with the nodes in that c-component
# add the directed parents of the nodes in that c-component
# remove V_i itself
# whatever remains is the conditioning set for P(V_i | ...)
def compute_c_components(
    graph: ADMGStructure,
    subset: tuple[str, ...] | list[str] | None = None,
) -> tuple[frozenset[str], ...]:
    active_nodes = tuple(subset) if subset is not None else graph.topological_order
    active_set = set(active_nodes)
    components: list[frozenset[str]] = []
    visited: set[str] = set()

    for node_name in active_nodes:
        if node_name in visited:
            continue
        stack = [node_name]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in bidirected_neighbors(graph, current):
                if neighbor in active_set and neighbor not in visited:
                    stack.append(neighbor)
        components.append(frozenset(component))

    return tuple(components)


def compute_prefix_c_components(graph: ADMGStructure) -> dict[str, tuple[frozenset[str], ...]]:
    prefix_components: dict[str, tuple[frozenset[str], ...]] = {}
    for index, node_name in enumerate(graph.topological_order, start=1):
        prefix_nodes = graph.topological_order[:index]
        prefix_components[node_name] = compute_c_components(graph, subset=prefix_nodes)
    return prefix_components


def derive_node_predecessor_sets(graph: ADMGStructure) -> dict[str, tuple[str, ...]]:
    prefix_components = compute_prefix_c_components(graph)
    predecessor_sets: dict[str, tuple[str, ...]] = {}

    for index, node_name in enumerate(graph.topological_order, start=1):
        prefix_nodes = graph.topological_order[:index]
        component = next(part for part in prefix_components[node_name] if node_name in part)

        effective_nodes = set(component)
        for member in component:
            member_index = graph.index_by_name[member]
            for parent_name in prefix_nodes:
                parent_index = graph.index_by_name[parent_name]
                if graph.directed_adj[parent_index, member_index]:
                    effective_nodes.add(parent_name)

        conditioning = tuple(
            candidate
            for candidate in prefix_nodes
            if candidate != node_name and candidate in effective_nodes
        )
        predecessor_sets[node_name] = conditioning

    return predecessor_sets


def _make_skeleton_rows(
    conditioned_on: tuple[str, ...],
    scope: tuple[str, ...],
    states_by_var: dict[str, tuple[Any, ...]],
    *,
    probability_placeholder: str,
) -> tuple[SkeletonTableRow, ...]:
    rows: list[SkeletonTableRow] = []
    for conditioning_values in assignment_space(conditioned_on, states_by_var):
        for scope_values in assignment_space(scope, states_by_var):
            rows.append(
                SkeletonTableRow(
                    conditioning_values=conditioning_values,
                    scope_values=scope_values,
                    probability_placeholder=probability_placeholder,
                )
            )
    return tuple(rows)


def build_node_kernel_skeletons(
    graph: ADMGStructure,
    *,
    probability_placeholder: str = PROBABILITY_PLACEHOLDER,
) -> dict[str, NodeKernelSkeleton]:
    predecessor_sets = derive_node_predecessor_sets(graph)
    kernels: dict[str, NodeKernelSkeleton] = {}

    for node_name in graph.topological_order:
        conditioned_on = predecessor_sets[node_name]
        kernels[node_name] = NodeKernelSkeleton(
            variable=node_name,
            conditioned_on=conditioned_on,
            states_by_var={var: graph.states_by_var[var] for var in (*conditioned_on, node_name)},
            rows=_make_skeleton_rows(
                conditioned_on,
                (node_name,),
                graph.states_by_var,
                probability_placeholder=probability_placeholder,
            ),
            probability_placeholder=probability_placeholder,
        )

    validate_node_kernel_skeletons(graph, kernels)
    return kernels


def validate_node_kernel_skeletons(
    graph: ADMGStructure,
    kernels: dict[str, NodeKernelSkeleton],
) -> None:
    if set(kernels) != set(graph.node_names):
        raise ValueError("Node kernel skeletons must be provided for every graph node.")

    expected_predecessors = derive_node_predecessor_sets(graph)
    for node_name in graph.topological_order:
        kernel = kernels[node_name]
        if kernel.variable != node_name:
            raise ValueError(f"Kernel key {node_name} does not match kernel variable {kernel.variable}.")
        if kernel.conditioned_on != expected_predecessors[node_name]:
            raise ValueError(
                f"{node_name} expected conditioning set {expected_predecessors[node_name]}, "
                f"got {kernel.conditioned_on}."
            )

        expected_row_count = len(assignment_space(kernel.conditioned_on, graph.states_by_var)) * len(
            graph.states_by_var[node_name]
        )
        if len(kernel.rows) != expected_row_count:
            raise ValueError(
                f"{node_name} expected {expected_row_count} skeleton rows, got {len(kernel.rows)}."
            )

        for row in kernel.rows:
            if row.probability_placeholder != kernel.probability_placeholder:
                raise ValueError(f"{node_name} uses inconsistent probability placeholders.")


def _kernel_probability_label(kernel: NodeKernelSkeleton) -> str:
    if kernel.conditioned_on:
        return f"P({kernel.variable} | {', '.join(kernel.conditioned_on)})"
    return f"P({kernel.variable})"


def _factor_name(scope: tuple[str, ...]) -> str:
    return f"Q[{', '.join(scope)}]"


def derive_tian_pearl_factor_skeletons(
    graph: ADMGStructure,
    kernels: dict[str, NodeKernelSkeleton],
    *,
    probability_placeholder: str = PROBABILITY_PLACEHOLDER,
) -> dict[str, TianPearlFactorSkeleton]:
    validate_node_kernel_skeletons(graph, kernels)
    factors: dict[str, TianPearlFactorSkeleton] = {}

    for component in compute_c_components(graph):
        scope = tuple(node_name for node_name in graph.topological_order if node_name in component)
        conditioned_on = tuple(
            node_name
            for node_name in graph.topological_order
            if node_name not in component
            and any(node_name in kernels[member].conditioned_on for member in scope)
        )
        name = _factor_name(scope)
        factors[name] = TianPearlFactorSkeleton(
            name=name,
            component=component,
            scope=scope,
            conditioned_on=conditioned_on,
            term_variables=scope,
            symbolic_factorization=f"{name} = " + " * ".join(
                _kernel_probability_label(kernels[variable]) for variable in scope
            ),
            states_by_var={var: graph.states_by_var[var] for var in (*conditioned_on, *scope)},
            rows=_make_skeleton_rows(
                conditioned_on,
                scope,
                graph.states_by_var,
                probability_placeholder=probability_placeholder,
            ),
            probability_placeholder=probability_placeholder,
        )

    return factors


def build_admg_from_config(
    *,
    node_names: tuple[str, ...],
    topological_order: tuple[str, ...],
    directed_adj: np.ndarray,
    bidirected_adj: np.ndarray,
    states_by_var: dict[str, tuple[Any, ...]],
    probability_placeholder: str = PROBABILITY_PLACEHOLDER,
) -> InitializedADMGModel:
    graph = ADMGStructure(
        node_names=node_names,
        topological_order=topological_order,
        directed_adj=directed_adj,
        bidirected_adj=bidirected_adj,
        states_by_var=states_by_var,
    )
    kernels = build_node_kernel_skeletons(
        graph,
        probability_placeholder=probability_placeholder,
    )
    factors = derive_tian_pearl_factor_skeletons(
        graph,
        kernels,
        probability_placeholder=probability_placeholder,
    )
    return InitializedADMGModel(graph=graph, node_kernels=kernels, factors=factors)


# USER-EDITABLE EXAMPLE CONFIG
EXAMPLE_NODE_NAMES = ("X1", "Z1", "Y1", "X2", "Z2", "Y2")
EXAMPLE_TOPOLOGICAL_ORDER = EXAMPLE_NODE_NAMES
EXAMPLE_DIRECTED_ADJ = np.array(
    [
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)
EXAMPLE_BIDIRECTED_ADJ = np.array(
    [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ],
    dtype=int,
)
EXAMPLE_STATES_BY_VAR = {node_name: BINARY_STATES for node_name in EXAMPLE_NODE_NAMES}


def build_example_model(
    *,
    probability_placeholder: str = PROBABILITY_PLACEHOLDER,
) -> InitializedADMGModel:
    return build_admg_from_config(
        node_names=EXAMPLE_NODE_NAMES,
        topological_order=EXAMPLE_TOPOLOGICAL_ORDER,
        directed_adj=EXAMPLE_DIRECTED_ADJ,
        bidirected_adj=EXAMPLE_BIDIRECTED_ADJ,
        states_by_var=EXAMPLE_STATES_BY_VAR,
        probability_placeholder=probability_placeholder,
    )


def build_frontdoor_two_slice_example(
    *,
    probability_placeholder: str = PROBABILITY_PLACEHOLDER,
) -> InitializedADMGModel:
    return build_example_model(probability_placeholder=probability_placeholder)


def _short_var_name(name: str) -> str:
    return name.replace("_{t-1}", "-").replace("_t", "t")


def _render_table_block(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    widths = []
    for column_index, header in enumerate(headers):
        max_row_width = max((len(row[column_index]) for row in rows), default=0)
        widths.append(max(len(header), max_row_width))

    def render_row(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[column_index]) for column_index, value in enumerate(values)
        )

    divider = "  ".join("-" * width for width in widths)
    lines = [title, render_row(headers), divider]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _format_kernel_skeleton_block(
    kernel: NodeKernelSkeleton,
    *,
    compact: bool,
) -> str:
    variable = _short_var_name(kernel.variable) if compact else kernel.variable
    headers = [
        *(_short_var_name(name) if compact else name for name in kernel.conditioned_on),
        variable,
        f"P({variable})",
    ]
    rows = [
        [
            *[str(value) for value in row.conditioning_values],
            str(row.scope_values[0]),
            row.probability_placeholder,
        ]
        for row in kernel.rows
    ]
    return _render_table_block(_kernel_probability_label(kernel), headers, rows)


def format_node_kernel_skeletons(
    kernels: dict[str, NodeKernelSkeleton],
    *,
    node_order: tuple[str, ...],
    compact: bool = False,
) -> str:
    return "\n\n".join(
        _format_kernel_skeleton_block(kernels[node_name], compact=compact)
        for node_name in node_order
    )


def _format_factor_skeleton_block(
    factor: TianPearlFactorSkeleton,
    *,
    compact: bool,
) -> str:
    headers = [
        *(_short_var_name(name) if compact else name for name in factor.conditioned_on),
        *(_short_var_name(name) if compact else name for name in factor.scope),
        factor.name,
    ]
    rows = [
        [
            *[str(value) for value in row.conditioning_values],
            *[str(value) for value in row.scope_values],
            row.probability_placeholder,
        ]
        for row in factor.rows
    ]
    return _render_table_block(factor.symbolic_factorization, headers, rows)


def format_tian_pearl_factor_skeletons(
    factors: dict[str, TianPearlFactorSkeleton],
    *,
    compact: bool = False,
) -> str:
    return "\n\n".join(
        _format_factor_skeleton_block(factor, compact=compact)
        for factor in factors.values()
    )


def summarize_admg_initializer(
    graph: ADMGStructure,
    kernels: dict[str, NodeKernelSkeleton],
    factors: dict[str, TianPearlFactorSkeleton],
) -> str:
    prefix_components = compute_prefix_c_components(graph)
    predecessor_sets = derive_node_predecessor_sets(graph)

    lines = [
        "Generic ADMG + Tian-Pearl skeleton builder",
        f"Nodes: {', '.join(graph.node_names)}",
        f"Topological order: {' < '.join(graph.topological_order)}",
        "Directed edges:",
    ]
    lines.extend(f"  {src} -> {dst}" for src, dst in graph.directed_edges)
    lines.append("Bidirected edges:")
    lines.extend(f"  {left} <-> {right}" for left, right in graph.bidirected_edges)
    lines.append("Full-graph c-components:")
    lines.extend(
        f"  {index + 1}. {{{', '.join(sorted(component))}}}"
        for index, component in enumerate(compute_c_components(graph))
    )
    lines.append("Prefix-derived node conditioning sets:")
    for node_name in graph.topological_order:
        prefix_label = ", ".join(
            "{" + ", ".join(sorted(component)) + "}"
            for component in prefix_components[node_name]
        )
        predecessors = predecessor_sets[node_name]
        predecessor_label = ", ".join(predecessors) if predecessors else "[]"
        lines.append(
            f"  {node_name}: conditioned_on={predecessor_label}; prefix components={prefix_label}"
        )
    lines.append(f"Kernel skeletons: {len(kernels)}")
    lines.append(f"Derived c-factor skeletons: {len(factors)}")
    return "\n".join(lines)


def _split_node_time_label(name: str) -> tuple[str, str | None]:
    if name.endswith("_{t-1}"):
        return name[: -len("_{t-1}")], "t-1"
    if name.endswith("_t"):
        return name[: -len("_t")], "t"

    match = re.fullmatch(r"(.+?)(\d+)", name)
    if match:
        return match.group(1), match.group(2)

    return name, None


def _build_graph_positions(graph: ADMGStructure) -> dict[str, tuple[float, float]]:
    parsed = {node_name: _split_node_time_label(node_name) for node_name in graph.topological_order}
    slice_labels = [slice_label for _, slice_label in parsed.values() if slice_label is not None]
    unique_slice_labels = list(dict.fromkeys(slice_labels))

    if slice_labels and len(slice_labels) == len(graph.node_names) and 1 < len(unique_slice_labels) <= 4:
        base_order = list(dict.fromkeys(base for base, _ in parsed.values()))
        slice_x = {slice_label: float(index) for index, slice_label in enumerate(unique_slice_labels)}
        base_y = {
            base_name: float(len(base_order) - 1 - index)
            for index, base_name in enumerate(base_order)
        }
        return {
            node_name: (slice_x[slice_label], base_y[base_name])
            for node_name, (base_name, slice_label) in parsed.items()
        }

    graph_view = nx.DiGraph()
    graph_view.add_nodes_from(graph.node_names)
    graph_view.add_edges_from(graph.directed_edges)
    initial_pos = {
        node_name: (index, -0.5 * index)
        for index, node_name in enumerate(graph.topological_order)
    }
    return nx.spring_layout(graph_view, seed=7, pos=initial_pos)


def _temporal_edges(graph: ADMGStructure) -> set[tuple[str, str]]:
    temporal: set[tuple[str, str]] = set()
    parsed = {node_name: _split_node_time_label(node_name) for node_name in graph.node_names}
    for src, dst in graph.directed_edges:
        src_base, src_slice = parsed[src]
        dst_base, dst_slice = parsed[dst]
        if src_base == dst_base and src_slice is not None and dst_slice is not None and src_slice != dst_slice:
            temporal.add((src, dst))
    return temporal


def _is_vertical_temporal_edge(
    positions: dict[str, tuple[float, float]],
    src: str,
    dst: str,
) -> bool:
    src_x, src_y = positions[src]
    dst_x, dst_y = positions[dst]
    return abs(src_y - dst_y) < 1e-9 and abs(src_x - dst_x) > 1e-9


def visualize_admg_initializer(
    graph: ADMGStructure,
    kernels: dict[str, NodeKernelSkeleton],
    factors: dict[str, TianPearlFactorSkeleton],
):
    fig = plt.figure(figsize=(21.0, 13.0))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.00, 1.15], wspace=0.10)

    ax_graph = fig.add_subplot(outer[0, 0])
    right = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0, 1],
        height_ratios=[0.34, 1.02, 0.38],
        hspace=0.20,
        wspace=0.16,
    )
    ax_directed = fig.add_subplot(right[0, 0])
    ax_bidirected = fig.add_subplot(right[0, 1])
    ax_kernels_left = fig.add_subplot(right[1, 0])
    ax_kernels_right = fig.add_subplot(right[1, 1])
    ax_factors = fig.add_subplot(right[2, :])

    positions = _build_graph_positions(graph)
    graph_view = nx.DiGraph()
    graph_view.add_nodes_from(graph.node_names)
    graph_view.add_edges_from(graph.directed_edges)
    temporal_edges = _temporal_edges(graph)

    palette = ["#54708b", "#2f9e8f", "#c77d4a", "#9c6644", "#6c757d", "#8d99ae"]
    color_by_node = {
        node_name: palette[index % len(palette)]
        for index, node_name in enumerate(graph.topological_order)
    }

    ax_graph.set_facecolor("#faf7f2")
    nx.draw_networkx_nodes(
        graph_view,
        positions,
        ax=ax_graph,
        node_color=[color_by_node[node] for node in graph.node_names],
        node_size=1800,
        edgecolors="#1f1f1f",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph_view,
        positions,
        ax=ax_graph,
        font_color="white",
        font_size=9,
        font_weight="bold",
    )

    def add_arrow(src: str, dst: str, *, color: str, linestyle: str = "-", rad: float = 0.0) -> None:
        patch = FancyArrowPatch(
            posA=positions[src],
            posB=positions[dst],
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.4,
            linestyle=linestyle,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=32,
            shrinkB=32,
            zorder=2,
        )
        ax_graph.add_patch(patch)

    for src, dst in graph.directed_edges:
        if (src, dst) in temporal_edges:
            temporal_rad = 0.18 if _is_vertical_temporal_edge(positions, src, dst) else 0.08
            add_arrow(src, dst, color="#1f1f1f", linestyle="-", rad=temporal_rad)
        else:
            add_arrow(src, dst, color="#1f1f1f")

    for edge_index, (left, right) in enumerate(graph.bidirected_edges):
        patch = FancyArrowPatch(
            posA=positions[left],
            posB=positions[right],
            arrowstyle="<->",
            mutation_scale=14,
            linewidth=1.8,
            linestyle="dotted",
            color="#b24c4c",
            connectionstyle=f"arc3,rad={0.20 + 0.05 * (edge_index % 2)}",
            zorder=1,
        )
        ax_graph.add_patch(patch)

    component_text = "\n".join(
        f"Q[{', '.join(component)}]"
        for component in compute_c_components(graph)
    )
    ax_graph.text(
        0.02,
        0.98,
        component_text,
        transform=ax_graph.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#4a4034",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#fff5dd", "edgecolor": "#d1b37d"},
    )
    ax_graph.text(
        0.50,
        -0.08,
        "Solid black arrows = directed edges, dotted red arcs = bidirected latent confounding",
        transform=ax_graph.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax_graph.set_title("Matrix-Driven ADMG Structure", fontsize=14, fontweight="bold", color="#222222", pad=12)
    ax_graph.axis("off")

    def draw_matrix(ax, matrix: np.ndarray, title: str) -> None:
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        labels = [_short_var_name(name) for name in graph.node_names]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    str(int(matrix[row_index, col_index])),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#0f172a" if matrix[row_index, col_index] == 0 else "white",
                )

    draw_matrix(ax_directed, graph.directed_adj, "Directed adjacency")
    draw_matrix(ax_bidirected, graph.bidirected_adj, "Bidirected adjacency")

    split_index = (len(graph.topological_order) + 1) // 2
    kernel_columns = (
        graph.topological_order[:split_index],
        graph.topological_order[split_index:],
    )
    for axis, node_subset in zip((ax_kernels_left, ax_kernels_right), kernel_columns):
        axis.set_facecolor("#fffaf1")
        axis.axis("off")
        axis.text(
            0.02,
            0.98,
            format_node_kernel_skeletons(kernels, node_order=tuple(node_subset), compact=True),
            ha="left",
            va="top",
            family="monospace",
            fontsize=8.0,
            color="#433728",
            transform=axis.transAxes,
        )
    ax_kernels_left.set_title("Node-local CPT skeleton", fontsize=11, fontweight="bold", loc="left")

    factor_lines = ["Derived Tian-Pearl c-factor skeletons"]
    for factor in factors.values():
        factor_lines.append(factor.symbolic_factorization)
        factor_lines.append(
            f"  columns=({', '.join((*factor.conditioned_on, *factor.scope, factor.name))})"
        )
        factor_lines.append(f"  rows={len(factor.rows)}, placeholder={factor.probability_placeholder}")
    ax_factors.set_facecolor("#f7f2e8")
    ax_factors.axis("off")
    ax_factors.text(
        0.01,
        0.98,
        "\n".join(factor_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.8,
        color="#2d261e",
        transform=ax_factors.transAxes,
    )

    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.05)
    return fig, ax_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and visualize a generic ADMG skeleton from user-editable matrices."
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).with_suffix(".png"),
        help="Path where the figure should be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    args = parser.parse_args()

    model = build_example_model()
    print(summarize_admg_initializer(model.graph, model.node_kernels, model.factors))
    print()
    print("Node-local CPT/kernel skeletons")
    print(format_node_kernel_skeletons(model.node_kernels, node_order=model.graph.topological_order))
    print()
    print("Derived Tian-Pearl c-factor skeletons")
    print(format_tian_pearl_factor_skeletons(model.factors))

    fig, _ = visualize_admg_initializer(model.graph, model.node_kernels, model.factors)
    fig.savefig(args.save, dpi=180, bbox_inches="tight")
    print(f"\nSaved figure to {args.save}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
