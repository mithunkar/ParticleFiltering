import numpy as np


N_PARTICLES = 100_000
SEED = 0

P_U1 = 0.5
P_X1_GIVEN_U = {0: 0.2, 1: 0.8}
P_Z1_GIVEN_X = {0: 0.15, 1: 0.85}
P_Y1_GIVEN_ZU = {
    (0, 0): 0.10,
    (1, 0): 0.60,
    (0, 1): 0.40,
    (1, 1): 0.90,
}


def normalize(dist):
    dist = np.asarray(dist, dtype=float)
    return dist / dist.sum()


def bernoulli(prob_one, rng):
    return int(rng.random() < prob_one)


def systematic_resample(particles, weights, rng):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Cannot resample with zero weight.")

    weights = weights / total
    positions = (np.arange(len(particles)) + rng.random()) / len(particles)
    cumsum = np.cumsum(weights)

    resampled = []
    i = 0
    j = 0
    while i < len(particles):
        if positions[i] < cumsum[j]:
            resampled.append(dict(particles[j]))
            i += 1
        else:
            j += 1

    return resampled


def sample_particle(rng):
    u = bernoulli(P_U1, rng)
    x = bernoulli(P_X1_GIVEN_U[u], rng)
    z = bernoulli(P_Z1_GIVEN_X[x], rng)
    y = bernoulli(P_Y1_GIVEN_ZU[(z, u)], rng)
    return {"U": u, "X": x, "Z": z, "Y": y}


def sample_until_x_stage(n_particles, rng):
    particles = []
    for _ in range(n_particles):
        u = bernoulli(P_U1, rng)
        x = bernoulli(P_X1_GIVEN_U[u], rng)
        particles.append({"U": u, "X": x})
    return particles


def propagate_to_z(particles, rng):
    out = []
    for particle in particles:
        updated = dict(particle)
        updated["Z"] = bernoulli(P_Z1_GIVEN_X[updated["X"]], rng)
        out.append(updated)
    return out


def propagate_to_y(particles, rng):
    out = []
    for particle in particles:
        updated = dict(particle)
        updated["Y"] = bernoulli(P_Y1_GIVEN_ZU[(updated["Z"], updated["U"])], rng)
        out.append(updated)
    return out


def condition_on(particles, variable, value, rng):
    weights = np.array([1.0 if p[variable] == value else 0.0 for p in particles])
    if weights.sum() == 0:
        raise ValueError(f"No particles matched {variable}={value}.")
    return systematic_resample(particles, weights, rng)


def binary_marginal(particles, variable):
    counts = np.zeros(2, dtype=float)
    for particle in particles:
        counts[particle[variable]] += 1.0
    return normalize(counts)


def estimate_p_x(n_particles=N_PARTICLES, seed=SEED):
    rng = np.random.default_rng(seed)
    particles = sample_until_x_stage(n_particles, rng)
    return binary_marginal(particles, "X")


def estimate_p_z_given_x(x_value, n_particles=N_PARTICLES, seed=SEED):
    rng = np.random.default_rng(seed)
    particles = sample_until_x_stage(n_particles, rng)
    particles = condition_on(particles, "X", x_value, rng)
    particles = propagate_to_z(particles, rng)
    return binary_marginal(particles, "Z")


def estimate_p_y_given_x_z(x_value, z_value, n_particles=N_PARTICLES, seed=SEED):
    rng = np.random.default_rng(seed)
    particles = sample_until_x_stage(n_particles, rng)
    particles = condition_on(particles, "X", x_value, rng)
    particles = propagate_to_z(particles, rng)
    particles = condition_on(particles, "Z", z_value, rng)
    particles = propagate_to_y(particles, rng)
    return binary_marginal(particles, "Y")


def estimate_frontdoor(x_value, n_particles=N_PARTICLES, seed=SEED):
    p_x = estimate_p_x(n_particles=n_particles, seed=seed)

    p_z_given_x = {}
    for z_value in (0, 1):
        p_z_given_x[z_value] = estimate_p_z_given_x(
            x_value,
            n_particles=n_particles,
            seed=seed + 10 + z_value,
        )

    p_y_given_x_z = {}
    for x_prime in (0, 1):
        for z_value in (0, 1):
            p_y_given_x_z[(x_prime, z_value)] = estimate_p_y_given_x_z(
                x_prime,
                z_value,
                n_particles=n_particles,
                seed=seed + 100 + 10 * x_prime + z_value,
            )

    result = np.zeros(2, dtype=float)
    for y_value in (0, 1):
        total = 0.0
        for z_value in (0, 1):
            inner = 0.0
            for x_prime in (0, 1):
                inner += p_y_given_x_z[(x_prime, z_value)][y_value] * p_x[x_prime]
            total += p_z_given_x[z_value][z_value] * inner
        result[y_value] = total

    return {
        "p_x": p_x,
        "p_z_given_x": p_z_given_x,
        "p_y_given_x_z": p_y_given_x_z,
        "frontdoor": normalize(result),
    }


def format_distribution(name, dist, labels):
    parts = [f"{label}: {prob:.6f}" for label, prob in zip(labels, dist)]
    return f"{name:<22} " + " | ".join(parts)


def run_experiment():
    p_x = estimate_p_x()

    p_z_given_x = {}
    for x_value in (0, 1):
        p_z_given_x[x_value] = estimate_p_z_given_x(x_value, seed=SEED + x_value)

    p_y_given_x_z = {}
    for x_value in (0, 1):
        for z_value in (0, 1):
            p_y_given_x_z[(x_value, z_value)] = estimate_p_y_given_x_z(
                x_value,
                z_value,
                seed=SEED + 20 + 10 * x_value + z_value,
            )

    frontdoor_by_x = {}
    for x_value in (0, 1):
        frontdoor_by_x[x_value] = estimate_frontdoor(x_value, seed=SEED + 200 + x_value)

    distributions = [p_x]
    distributions.extend(p_z_given_x.values())
    distributions.extend(p_y_given_x_z.values())
    distributions.extend(frontdoor_by_x[x]["frontdoor"] for x in (0, 1))
    for dist in distributions:
        assert np.isclose(dist.sum(), 1.0, atol=1e-9)

    assert p_z_given_x[1][1] > p_z_given_x[0][1] + 0.6
    assert frontdoor_by_x[1]["frontdoor"][1] > frontdoor_by_x[0]["frontdoor"][1]

    repeat = estimate_frontdoor(1, seed=SEED + 201)
    assert np.allclose(repeat["frontdoor"], frontdoor_by_x[1]["frontdoor"])

    manual = []
    for y_value in (0, 1):
        total = 0.0
        for z_value in (0, 1):
            inner = 0.0
            for x_prime in (0, 1):
                inner += (
                    frontdoor_by_x[1]["p_y_given_x_z"][(x_prime, z_value)][y_value]
                    * frontdoor_by_x[1]["p_x"][x_prime]
                )
            total += frontdoor_by_x[1]["p_z_given_x"][z_value][z_value] * inner
        manual.append(total)
    manual = normalize(manual)
    assert np.allclose(manual, frontdoor_by_x[1]["frontdoor"])

    print("Frontdoor experiment with particle filtering")
    print(f"particles={N_PARTICLES:,}, seed={SEED}")
    print()
    print("P(X)")
    print(format_distribution("P(X)", p_x, ("X=0", "X=1")))
    print()
    print("P(Z | X=x)")
    for x_value in (0, 1):
        print(format_distribution(f"P(Z | X={x_value})", p_z_given_x[x_value], ("Z=0", "Z=1")))
    print()
    print("P(Y | X=x', Z=z)")
    for x_value in (0, 1):
        for z_value in (0, 1):
            print(
                format_distribution(
                    f"P(Y | X={x_value}, Z={z_value})",
                    p_y_given_x_z[(x_value, z_value)],
                    ("Y=0", "Y=1"),
                )
            )
    print()
    print("P(Y | do(X=x))")
    for x_value in (0, 1):
        print(
            format_distribution(
                f"P(Y | do(X={x_value}))",
                frontdoor_by_x[x_value]["frontdoor"],
                ("Y=0", "Y=1"),
            )
        )
    print()
    print("Checks passed.")


if __name__ == "__main__":
    run_experiment()
