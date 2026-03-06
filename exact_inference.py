"""
Exact Inference for Dynamic Bayesian Networks with Interventions

Uses pgmpy library for exact inference via variable elimination.
Supports interventional queries: P(Query | do(Interventions), Evidence)

Problem specification from problem.txt:
- Binary variables: W (Weather), T (Temp), E (Energy), U (Umbrella), A (Activity)
- Dependencies:
  - W_t | W_{t-1}
  - T_t | T_{t-1}, W_{t-1}, W_t
  - E_t | E_{t-1}, W_t, T_t
  - U_t | W_t (sensor)
  - A_t | E_t, W_t (sensor)

Example query: P(A_10 | do(W_5, T_5, E_5), U_{5,6,7}, A_{5,6,7})
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from collections import defaultdict
import time


def build_binary_dbn(T_max: int = 10):
    """
    Build a DBN as an "unrolled" Bayesian Network up to time T_max.
    Binary variables only for simplicity.
    
    Variables per timestep:
    - W_t: Weather (0=sunny, 1=cloudy)
    - T_t: Temperature (0=hot, 1=cold)
    - E_t: Energy (0=high, 1=low)
    - U_t: Umbrella (0=no, 1=yes)
    - A_t: Activity (0=outdoor, 1=indoor)
    
    Returns:
        DiscreteBayesianNetwork: Unrolled DBN
        dict: CPD templates for reuse
    """
    
    # Define network structure (edges)
    edges = []
    
    for t in range(T_max):
        # Within-slice edges (same time)
        edges.append((f'W_{t}', f'U_{t}'))      # W_t -> U_t
        edges.append((f'E_{t}', f'A_{t}'))      # E_t -> A_t
        edges.append((f'W_{t}', f'A_{t}'))      # W_t -> A_t
        edges.append((f'W_{t}', f'T_{t}'))      # W_t -> T_t
        edges.append((f'W_{t}', f'E_{t}'))      # W_t -> E_t
        edges.append((f'T_{t}', f'E_{t}'))      # T_t -> E_t
        
        # Cross-slice edges (temporal)
        if t > 0:
            edges.append((f'W_{t-1}', f'W_{t}'))    # W_{t-1} -> W_t
            edges.append((f'W_{t-1}', f'T_{t}'))    # W_{t-1} -> T_t
            edges.append((f'T_{t-1}', f'T_{t}'))    # T_{t-1} -> T_t
            edges.append((f'E_{t-1}', f'E_{t}'))    # E_{t-1} -> E_t
    
    # Create Bayesian Network
    model = DiscreteBayesianNetwork(edges)
    
    # Define CPDs (Conditional Probability Distributions)
    cpds = []
    
    # ══════════════════════════════════════════════════════════
    # Time slice 0: Priors
    # ══════════════════════════════════════════════════════════
    
    # P(W_0): Weather prior
    cpd_w0 = TabularCPD(
        variable='W_0',
        variable_card=2,
        values=[[0.6],   # P(W_0=sunny)
                [0.4]]   # P(W_0=cloudy)
    )
    cpds.append(cpd_w0)
    
    # P(T_0 | W_0): Temperature prior given weather
    cpd_t0 = TabularCPD(
        variable='T_0',
        variable_card=2,
        values=[
            [0.7, 0.3],  # P(T_0=hot | W_0=sunny/cloudy)
            [0.3, 0.7]   # P(T_0=cold | W_0=sunny/cloudy)
        ],
        evidence=['W_0'],
        evidence_card=[2]
    )
    cpds.append(cpd_t0)
    
    # P(E_0 | W_0, T_0): Energy prior
    cpd_e0 = TabularCPD(
        variable='E_0',
        variable_card=2,
        values=[
            # W_0=sunny        W_0=cloudy
            # T_0: hot  cold   hot  cold
            [0.8, 0.6,   0.5, 0.3],  # P(E_0=high | ...)
            [0.2, 0.4,   0.5, 0.7]   # P(E_0=low | ...)
        ],
        evidence=['W_0', 'T_0'],
        evidence_card=[2, 2]
    )
    cpds.append(cpd_e0)
    
    # P(U_0 | W_0): Umbrella sensor
    cpd_u0 = TabularCPD(
        variable='U_0',
        variable_card=2,
        values=[
            [0.95, 0.2],  # P(U_0=no | W_0=sunny/cloudy)
            [0.05, 0.8]   # P(U_0=yes | W_0=sunny/cloudy)
        ],
        evidence=['W_0'],
        evidence_card=[2]
    )
    cpds.append(cpd_u0)
    
    # P(A_0 | E_0, W_0): Activity sensor
    cpd_a0 = TabularCPD(
        variable='A_0',
        variable_card=2,
        values=[
            # E_0=high         E_0=low
            # W_0: sunny cloudy sunny cloudy
            [0.9, 0.6,   0.5, 0.2],  # P(A_0=outdoor | ...)
            [0.1, 0.4,   0.5, 0.8]   # P(A_0=indoor | ...)
        ],
        evidence=['E_0', 'W_0'],
        evidence_card=[2, 2]
    )
    cpds.append(cpd_a0)
    
    # ══════════════════════════════════════════════════════════
    # Time slices t > 0: Transitions
    # ══════════════════════════════════════════════════════════
    
    for t in range(1, T_max):
        # P(W_t | W_{t-1}): Weather transition
        cpd_w = TabularCPD(
            variable=f'W_{t}',
            variable_card=2,
            values=[
                [0.7, 0.3],  # P(W_t=sunny | W_{t-1}=sunny/cloudy)
                [0.3, 0.7]   # P(W_t=cloudy | W_{t-1}=sunny/cloudy)
            ],
            evidence=[f'W_{t-1}'],
            evidence_card=[2]
        )
        cpds.append(cpd_w)
        
        # P(T_t | T_{t-1}, W_{t-1}, W_t): Temperature transition
        cpd_t = TabularCPD(
            variable=f'T_{t}',
            variable_card=2,
            values=[
                # T_{t-1}=hot              T_{t-1}=cold
                # W_{t-1}: sun  cloud      sun  cloud
                # W_t:     s c  s c        s c  s c
                [0.8,0.5, 0.6,0.4,    0.5,0.3, 0.3,0.2],  # P(T_t=hot | ...)
                [0.2,0.5, 0.4,0.6,    0.5,0.7, 0.7,0.8]   # P(T_t=cold | ...)
            ],
            evidence=[f'T_{t-1}', f'W_{t-1}', f'W_{t}'],
            evidence_card=[2, 2, 2]
        )
        cpds.append(cpd_t)
        
        # P(E_t | E_{t-1}, W_t, T_t): Energy transition
        cpd_e = TabularCPD(
            variable=f'E_{t}',
            variable_card=2,
            values=[
                # E_{t-1}=high                              E_{t-1}=low
                # W_t:     sunny    cloudy                  sunny    cloudy
                # T_t:     hot cold hot cold                hot cold hot cold
                [0.8,0.7, 0.6,0.4,                   0.6,0.4, 0.3,0.2],  # P(E_t=high | ...)
                [0.2,0.3, 0.4,0.6,                   0.4,0.6, 0.7,0.8]   # P(E_t=low | ...)
            ],
            evidence=[f'E_{t-1}', f'W_{t}', f'T_{t}'],
            evidence_card=[2, 2, 2]
        )
        cpds.append(cpd_e)
        
        # P(U_t | W_t): Umbrella sensor
        cpd_u = TabularCPD(
            variable=f'U_{t}',
            variable_card=2,
            values=[
                [0.95, 0.2],  # P(U_t=no | W_t=sunny/cloudy)
                [0.05, 0.8]   # P(U_t=yes | W_t=sunny/cloudy)
            ],
            evidence=[f'W_{t}'],
            evidence_card=[2]
        )
        cpds.append(cpd_u)
        
        # P(A_t | E_t, W_t): Activity sensor
        cpd_a = TabularCPD(
            variable=f'A_{t}',
            variable_card=2,
            values=[
                # E_t=high         E_t=low
                # W_t: sunny cloudy sunny cloudy
                [0.9, 0.6,   0.5, 0.2],  # P(A_t=outdoor | ...)
                [0.1, 0.4,   0.5, 0.8]   # P(A_t=indoor | ...)
            ],
            evidence=[f'E_{t}', f'W_{t}'],
            evidence_card=[2, 2]
        )
        cpds.append(cpd_a)
    
    # Add all CPDs to model
    model.add_cpds(*cpds)
    
    # Verify model is valid
    assert model.check_model()
    print(f"✓ Built valid DBN with {len(model.nodes())} nodes and {len(model.edges())} edges")
    
    return model


def intervene(model: DiscreteBayesianNetwork, interventions: dict) -> DiscreteBayesianNetwork:
    """
    Apply causal interventions to the model (do-operator).
    
    For each intervened variable:
    1. Remove all incoming edges (break causal links)
    2. Replace CPD with deterministic distribution at the intervention value
    
    Args:
        model: Original Bayesian Network
        interventions: dict mapping variable names to values, e.g., {'W_5': 0, 'T_5': 1}
    
    Returns:
        New DiscreteBayesianNetwork with interventions applied
    """
    # Create a copy of the model
    intervened_model = DiscreteBayesianNetwork()
    intervened_model.add_nodes_from(model.nodes())
    intervened_model.add_edges_from(model.edges())
    
    # Copy all CPDs first
    for cpd in model.get_cpds():
        intervened_model.add_cpds(cpd)
    
    # Apply interventions
    for var, value in interventions.items():
        # Remove incoming edges to break causal links
        parents = list(intervened_model.get_parents(var))
        for parent in parents:
            intervened_model.remove_edge(parent, var)
        
        # Create deterministic CPD: P(var = value) = 1
        intervened_cpd = TabularCPD(
            variable=var,
            variable_card=2,
            values=[[1.0 - value], [value]]  # P(var=0), P(var=1)
        )
        
        # Replace CPD
        intervened_model.remove_cpds(var)
        intervened_model.add_cpds(intervened_cpd)
    
    assert intervened_model.check_model()
    return intervened_model


def exact_inference_query(model: DiscreteBayesianNetwork, 
                          query_vars: list[str],
                          evidence: dict = None,
                          interventions: dict = None) -> dict:
    """
    Perform exact inference query with optional interventions.
    
    Computes: P(Query | do(Interventions), Evidence)
    
    Args:
        model: Bayesian Network (unrolled DBN)
        query_vars: List of variables to query
        evidence: dict of observed variables, e.g., {'U_5': 1, 'A_5': 0}
        interventions: dict of interventions, e.g., {'W_5': 0, 'T_5': 1}
    
    Returns:
        dict mapping query variable values to probabilities
    """
    evidence = evidence or {}
    interventions = interventions or {}
    
    # Apply interventions if any
    if interventions:
        print(f"\n[Intervention] Applying do({interventions})")
        model = intervene(model, interventions)
    
    # Perform variable elimination
    inference = VariableElimination(model)
    
    print(f"[Query] Computing P({query_vars} | evidence={evidence})")
    start_time = time.time()
    
    result = inference.query(
        variables=query_vars,
        evidence=evidence,
        show_progress=False
    )
    
    elapsed = time.time() - start_time
    print(f"[Result] Exact inference completed in {elapsed:.4f} seconds")
    
    return result, elapsed


def compare_interventions_example():
    """
    Example: Compare observation vs intervention
    
    Query: P(A_5 | U_5=yes)  vs  P(A_5 | do(W_5=cloudy))
    """
    print("\n" + "="*80)
    print("EXAMPLE: Observation vs Intervention")
    print("="*80)
    
    model = build_binary_dbn(T_max=6)
    
    # Observational query: P(A_5 | U_5=yes)
    print("\n--- Observational Query: P(A_5 | U_5=yes) ---")
    result_obs, _ = exact_inference_query(
        model,
        query_vars=['A_5'],
        evidence={'U_5': 1}  # observe umbrella=yes
    )
    print(result_obs)
    
    # Interventional query: P(A_5 | do(W_5=cloudy))
    print("\n--- Interventional Query: P(A_5 | do(W_5=cloudy)) ---")
    result_int, _ = exact_inference_query(
        model,
        query_vars=['A_5'],
        interventions={'W_5': 1}  # force weather=cloudy
    )
    print(result_int)


def problem_query_example():
    """
    Example from problem.txt:
    P(A_10 | do(W_5, T_5, E_5), U_{5,6,7}, A_{5,6,7})
    """
    print("\n" + "="*80)
    print("PROBLEM QUERY: P(A_10 | do(W_5=cloudy, T_5=cold, E_5=high), U_{5,6,7}, A_{5,6,7})")
    print("="*80)
    
    model = build_binary_dbn(T_max=11)  # Need up to time 10
    
    # Define interventions and evidence
    interventions = {
        'W_5': 1,  # cloudy
        'T_5': 1,  # cold
        'E_5': 0   # high
    }
    
    evidence = {
        'U_5': 1, 'U_6': 1, 'U_7': 0,  # umbrella observations
        'A_5': 1, 'A_6': 1, 'A_7': 0   # activity observations
    }
    
    result, elapsed = exact_inference_query(
        model,
        query_vars=['A_10'],
        evidence=evidence,
        interventions=interventions
    )
    
    print(result)
    print(f"\nProbability distribution:")
    print(f"  P(A_10=outdoor) = {result.values[0]:.6f}")
    print(f"  P(A_10=indoor)  = {result.values[1]:.6f}")


def marginal_over_time():
    """
    Track how belief about a variable evolves over time.
    Example: P(W_t) for t in [0, 10]
    """
    print("\n" + "="*80)
    print("MARGINAL PROBABILITIES OVER TIME: P(W_t)")
    print("="*80)
    
    T_max = 10
    model = build_binary_dbn(T_max=T_max + 1)
    inference = VariableElimination(model)
    
    print(f"\n{'Time':<6} {'P(sunny)':<12} {'P(cloudy)':<12}")
    print("-" * 35)
    
    for t in range(T_max + 1):
        result = inference.query(variables=[f'W_{t}'], show_progress=False)
        p_sunny = result.values[0]
        p_cloudy = result.values[1]
        print(f"{t:<6} {p_sunny:<12.6f} {p_cloudy:<12.6f}")


if __name__ == '__main__':
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "EXACT INFERENCE FOR DBN WITH INTERVENTIONS" + " "*16 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run examples
    compare_interventions_example()
    problem_query_example()
    marginal_over_time()
    
    print("\n" + "="*80)
    print("✓ All examples completed successfully!")
    print("="*80)
