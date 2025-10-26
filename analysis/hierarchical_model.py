import math
import os
import random
from typing import List, Tuple, Dict


def load_school_data(data_dir: str) -> List[List[float]]:
    groups: List[List[float]] = []
    for idx in range(1, 9):
        path = os.path.join(data_dir, f"school{idx}.dat")
        with open(path, "r", encoding="utf-8") as fh:
            values = [float(line.strip()) for line in fh if line.strip()]
        groups.append(values)
    return groups


def sample_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_variance(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_val = sample_mean(values)
    return sum((val - mean_val) ** 2 for val in values) / (n - 1)


def population_variance(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean_val = sample_mean(values)
    return sum((val - mean_val) ** 2 for val in values) / n


def quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    pos = q * (len(sorted_vals) - 1)
    lower_index = int(math.floor(pos))
    upper_index = int(math.ceil(pos))
    if lower_index == upper_index:
        return sorted_vals[lower_index]
    weight = pos - lower_index
    return sorted_vals[lower_index] * (1 - weight) + sorted_vals[upper_index] * weight


def standard_deviation(values: List[float]) -> float:
    return math.sqrt(population_variance(values))


def kernel_density_estimate(samples: List[float], grid: List[float]) -> List[float]:
    n = len(samples)
    if n == 0:
        return [0.0 for _ in grid]
    std = standard_deviation(samples)
    if std == 0:
        std = 1e-6
    bandwidth = 1.06 * std * n ** (-1 / 5)
    if bandwidth <= 0:
        bandwidth = 1e-3
    norm_const = 1.0 / (math.sqrt(2 * math.pi) * bandwidth * n)
    densities: List[float] = []
    for x in grid:
        total = 0.0
        for s in samples:
            z = (x - s) / bandwidth
            total += math.exp(-0.5 * z * z)
        densities.append(norm_const * total)
    return densities


def normal_pdf(x: float, mean: float, variance: float) -> float:
    std = math.sqrt(variance)
    return (1.0 / (math.sqrt(2 * math.pi) * std)) * math.exp(-0.5 * ((x - mean) / std) ** 2)


def inverse_gamma_pdf(x: float, shape: float, scale: float) -> float:
    if x <= 0:
        return 0.0
    return (scale ** shape) / math.gamma(shape) * (x ** (-shape - 1)) * math.exp(-scale / x)


def effective_sample_size(samples: List[float]) -> float:
    n = len(samples)
    if n < 2:
        return float("nan")
    mean_val = sample_mean(samples)
    variance = population_variance(samples)
    if variance == 0:
        return float("inf")
    denom = variance * n
    ac_sums = 0.0
    for lag in range(1, n):
        cov = sum((samples[i] - mean_val) * (samples[i + lag] - mean_val) for i in range(0, n - lag)) / n
        rho = cov / variance
        if rho <= 0:
            break
        ac_sums += 2 * rho
    return n / (1 + ac_sums)


def compute_group_statistics(groups: List[List[float]]) -> Tuple[List[int], List[float], List[float]]:
    n_list: List[int] = []
    means: List[float] = []
    variances: List[float] = []
    for values in groups:
        n_list.append(len(values))
        means.append(sample_mean(values))
        variances.append(sample_variance(values) if len(values) > 1 else 0.0)
    return n_list, means, variances


def gibbs_sampler(
    groups: List[List[float]],
    mu0: float,
    gamma0_sq: float,
    tau0_sq: float,
    eta0: float,
    sigma0_sq: float,
    nu0: float,
    iterations: int,
    burn_in: int,
    seed: int = 2024,
) -> Dict[str, List]:
    rng = random.Random(seed)
    m = len(groups)
    n_list, means, variances = compute_group_statistics(groups)

    theta = means[:]
    sigma_sq = sum(variances) / len(variances)
    if sigma_sq <= 0:
        sigma_sq = 1.0
    mu = sample_mean(theta)
    tau_sq = population_variance(theta)
    if tau_sq <= 0:
        tau_sq = tau0_sq

    theta_samples: List[List[float]] = []
    mu_samples: List[float] = []
    sigma_sq_samples: List[float] = []
    tau_sq_samples: List[float] = []

    for it in range(iterations):
        for j in range(m):
            n_j = n_list[j]
            mean_j = means[j]
            precision = n_j / sigma_sq + 1.0 / tau_sq
            variance_theta = 1.0 / precision
            mean_theta = variance_theta * (n_j * mean_j / sigma_sq + mu / tau_sq)
            theta[j] = rng.gauss(mean_theta, math.sqrt(variance_theta))

        total_n = sum(n_list)
        shape_sigma = 0.5 * (nu0 + total_n)
        sse = 0.0
        for j, values in enumerate(groups):
            theta_j = theta[j]
            sse += sum((val - theta_j) ** 2 for val in values)
        scale_sigma = 0.5 * (nu0 * sigma0_sq + sse)
        inv_sigma_sq = rng.gammavariate(shape_sigma, 1.0 / scale_sigma)
        sigma_sq = 1.0 / inv_sigma_sq

        precision_mu = m / tau_sq + 1.0 / gamma0_sq
        variance_mu = 1.0 / precision_mu
        mean_mu = variance_mu * (m * sample_mean(theta) / tau_sq + mu0 / gamma0_sq)
        mu = rng.gauss(mean_mu, math.sqrt(variance_mu))

        shape_tau = 0.5 * (eta0 + m)
        ss_tau = sum((theta_j - mu) ** 2 for theta_j in theta)
        scale_tau = 0.5 * (eta0 * tau0_sq + ss_tau)
        inv_tau_sq = rng.gammavariate(shape_tau, 1.0 / scale_tau)
        tau_sq = 1.0 / inv_tau_sq

        if it >= burn_in:
            theta_samples.append(theta[:])
            mu_samples.append(mu)
            sigma_sq_samples.append(sigma_sq)
            tau_sq_samples.append(tau_sq)

    return {
        "theta": theta_samples,
        "mu": mu_samples,
        "sigma_sq": sigma_sq_samples,
        "tau_sq": tau_sq_samples,
    }


def make_grid(min_val: float, max_val: float, num: int) -> List[float]:
    if max_val <= min_val:
        max_val = min_val + 1e-6
    step = (max_val - min_val) / (num - 1)
    return [min_val + i * step for i in range(num)]


def write_svg_line_plot(
    filename: str,
    series: List[List[float]],
    colors: List[str],
    labels: List[str],
    title: str,
    y_label: str,
    width: int = 900,
    height: int = 300,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    padding = 50
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding

    max_len = max(len(s) for s in series)
    x_values = list(range(max_len))

    all_values = [val for s in series for val in s]
    min_val = min(all_values)
    max_val = max(all_values)
    if max_val == min_val:
        max_val += 1.0
        min_val -= 1.0

    def scale_x(idx: int, length: int) -> float:
        if length <= 1:
            return padding
        return padding + (idx / (length - 1)) * plot_width

    def scale_y(value: float) -> float:
        return padding + (max_val - value) / (max_val - min_val) * plot_height

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>",
        f"<text x='{width / 2}' y='25' text-anchor='middle' font-size='16'>{title}</text>",
        f"<text x='{padding / 2}' y='{height / 2}' transform='rotate(-90 {padding / 2},{height / 2})' font-size='12'>{y_label}</text>",
        f"<line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='black' stroke-width='1'/>",
        f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='black' stroke-width='1'/>",
    ]

    for idx, s in enumerate(series):
        points = []
        length = len(s)
        for i, value in enumerate(s):
            x = scale_x(i, length)
            y = scale_y(value)
            points.append(f"{x:.2f},{y:.2f}")
        svg_lines.append(
            f"<polyline fill='none' stroke='{colors[idx]}' stroke-width='1.5' points='{' '.join(points)}'/>"
        )
        legend_x = padding + idx * 120
        svg_lines.append(
            f"<rect x='{legend_x}' y='{height - padding + 15}' width='12' height='12' fill='{colors[idx]}'/>"
        )
        svg_lines.append(
            f"<text x='{legend_x + 20}' y='{height - padding + 25}' font-size='12'>{labels[idx]}</text>"
        )

    svg_lines.append("</svg>")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg_lines))


def write_svg_density_plot(
    filename: str,
    grid: List[float],
    posterior_density: List[float],
    prior_density: List[float],
    title: str,
    x_label: str,
    legend_labels: Tuple[str, str],
    width: int = 900,
    height: int = 300,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    padding = 60
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding

    min_x = grid[0]
    max_x = grid[-1]
    max_y = max(posterior_density + prior_density)
    if max_y == 0:
        max_y = 1.0

    def scale_x(value: float) -> float:
        return padding + (value - min_x) / (max_x - min_x) * plot_width if max_x > min_x else padding

    def scale_y(value: float) -> float:
        return padding + (max_y - value) / max_y * plot_height

    def create_path(density: List[float], color: str) -> str:
        points = []
        for x_val, y_val in zip(grid, density):
            x = scale_x(x_val)
            y = scale_y(y_val)
            points.append(f"{x:.2f},{y:.2f}")
        return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(points)}'/>"

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect x='0' y='0' width='{0}' height='{1}' fill='white' stroke='none'/>".format(width, height),
        f"<text x='{width / 2}' y='30' text-anchor='middle' font-size='16'>{title}</text>",
        f"<text x='{width / 2}' y='{height - 10}' text-anchor='middle' font-size='12'>{x_label}</text>",
        f"<line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='black' stroke-width='1'/>",
        f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='black' stroke-width='1'/>",
    ]

    svg_lines.append(create_path(prior_density, "#888888"))
    svg_lines.append(create_path(posterior_density, "#1f77b4"))

    legend_x = padding
    legend_y = padding - 25
    svg_lines.append(
        f"<rect x='{legend_x}' y='{legend_y}' width='12' height='12' fill='#888888'/>"
    )
    svg_lines.append(
        f"<text x='{legend_x + 18}' y='{legend_y + 10}' font-size='12'>{legend_labels[0]}</text>"
    )
    svg_lines.append(
        f"<rect x='{legend_x + 180}' y='{legend_y}' width='12' height='12' fill='#1f77b4'/>"
    )
    svg_lines.append(
        f"<text x='{legend_x + 198}' y='{legend_y + 10}' font-size='12'>{legend_labels[1]}</text>"
    )

    svg_lines.append("</svg>")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg_lines))


def summarize_parameters(samples: Dict[str, List], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in ("sigma_sq", "mu", "tau_sq"):
        values = samples[key]
        summary[key] = {
            "mean": sample_mean(values),
            "lower": quantile(values, alpha / 2),
            "upper": quantile(values, 1 - alpha / 2),
            "ess": effective_sample_size(values),
        }
    return summary


def compute_R_samples(sigma_sq_samples: List[float], tau_sq_samples: List[float]) -> List[float]:
    r_samples: List[float] = []
    for sigma_sq, tau_sq in zip(sigma_sq_samples, tau_sq_samples):
        r_samples.append(tau_sq / (sigma_sq + tau_sq))
    return r_samples


def sample_prior_R(
    draws: int,
    eta0: float,
    tau0_sq: float,
    nu0: float,
    sigma0_sq: float,
    seed: int = 111,
) -> List[float]:
    rng = random.Random(seed)
    samples: List[float] = []
    shape_tau = eta0 / 2.0
    scale_tau = eta0 * tau0_sq / 2.0
    shape_sigma = nu0 / 2.0
    scale_sigma = nu0 * sigma0_sq / 2.0
    for _ in range(draws):
        inv_tau = rng.gammavariate(shape_tau, 1.0 / scale_tau)
        inv_sigma = rng.gammavariate(shape_sigma, 1.0 / scale_sigma)
        tau_sq = 1.0 / inv_tau
        sigma_sq = 1.0 / inv_sigma
        samples.append(tau_sq / (sigma_sq + tau_sq))
    return samples


def write_svg_scatter_plot(
    filename: str,
    x_values: List[float],
    y_values: List[float],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 600,
    height: int = 400,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    padding = 60
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    if max_x == min_x:
        max_x += 1
        min_x -= 1
    if max_y == min_y:
        max_y += 1
        min_y -= 1

    def scale_x(value: float) -> float:
        return padding + (value - min_x) / (max_x - min_x) * plot_width

    def scale_y(value: float) -> float:
        return padding + (max_y - value) / (max_y - min_y) * plot_height

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>",
        f"<text x='{width / 2}' y='30' text-anchor='middle' font-size='16'>{title}</text>",
        f"<text x='{width / 2}' y='{height - 15}' text-anchor='middle' font-size='12'>{x_label}</text>",
        f"<text x='{15}' y='{height / 2}' transform='rotate(-90 {15},{height / 2})' font-size='12'>{y_label}</text>",
        f"<line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='black' stroke-width='1'/>",
        f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='black' stroke-width='1'/>",
    ]

    for x_val, y_val in zip(x_values, y_values):
        x = scale_x(x_val)
        y = scale_y(y_val)
        svg_lines.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4' fill='#1f77b4' stroke='none'/>"
        )

    min_diag = max(min(min_x, min_y), min_x)
    max_diag = min(max(max_x, max_y), max_x)
    diag_points = []
    for frac in [i / 20 for i in range(21)]:
        val = min_diag + frac * (max_diag - min_diag)
        diag_points.append(f"{scale_x(val):.2f},{scale_y(val):.2f}")
    svg_lines.append(
        f"<polyline fill='none' stroke='#ff7f0e' stroke-width='1.5' points='{' '.join(diag_points)}'/>"
    )

    svg_lines.append("</svg>")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg_lines))


def write_summary_text(filename: str, summary: Dict[str, Dict[str, float]], ess: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = ["Parameter,Mean,Lower,Upper,ESS"]
    for key in ("sigma_sq", "mu", "tau_sq"):
        info = summary[key]
        lines.append(
            f"{key},{info['mean']:.6f},{info['lower']:.6f},{info['upper']:.6f},{info['ess']:.2f}"
        )
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_additional_metrics(
    filename: str,
    prob_theta7_lt_theta6: float,
    prob_theta7_smallest: float,
    sample_global_mean: float,
    posterior_mu_mean: float,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = [
        "metric,value",
        f"P(theta7 < theta6),{prob_theta7_lt_theta6:.4f}",
        f"P(theta7 smallest),{prob_theta7_smallest:.4f}",
        f"Sample overall mean,{sample_global_mean:.4f}",
        f"Posterior mean mu,{posterior_mu_mean:.4f}",
    ]
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def posterior_expectations(theta_samples: List[List[float]]) -> List[float]:
    if not theta_samples:
        return []
    m = len(theta_samples[0])
    sums = [0.0] * m
    for draw in theta_samples:
        for idx, value in enumerate(draw):
            sums[idx] += value
    total_draws = len(theta_samples)
    return [val / total_draws for val in sums]


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]


def summarize_mcmc(
    data_dir: str,
    output_dir: str,
    iterations: int = 60000,
    burn_in: int = 10000,
) -> Dict[str, any]:
    groups = load_school_data(data_dir)
    samples = gibbs_sampler(
        groups,
        mu0=7.0,
        gamma0_sq=5.0,
        tau0_sq=10.0,
        eta0=2.0,
        sigma0_sq=15.0,
        nu0=2.0,
        iterations=iterations,
        burn_in=burn_in,
    )

    summary = summarize_parameters(samples)
    ess_values = {key: summary[key]["ess"] for key in summary}

    traces = [samples["sigma_sq"], samples["mu"], samples["tau_sq"]]
    write_svg_line_plot(
        os.path.join(output_dir, "trace_sigma_mu_tau.svg"),
        traces,
        ["#2ca02c", "#1f77b4", "#d62728"],
        ["sigma^2", "mu", "tau^2"],
        "Trace Plots",
        "Value",
    )

    grids = {}
    density_data = {}
    for key, prior_params in (
        ("sigma_sq", (2.0 / 2.0, 2.0 * 15.0 / 2.0)),
        ("mu", None),
        ("tau_sq", (2.0 / 2.0, 2.0 * 10.0 / 2.0)),
    ):
        samples_key = samples[key]
        min_val = min(samples_key)
        max_val = max(samples_key)
        span = max_val - min_val
        grid = make_grid(min_val - 0.25 * span, max_val + 0.25 * span, 200)
        posterior_density = kernel_density_estimate(samples_key, grid)
        if key == "mu":
            prior_density = [normal_pdf(x, 7.0, 5.0) for x in grid]
        else:
            shape, scale = prior_params
            prior_density = [inverse_gamma_pdf(x, shape, scale) for x in grid]
        density_data[key] = (grid, posterior_density, prior_density)

    write_svg_density_plot(
        os.path.join(output_dir, "density_sigma_sq.svg"),
        *density_data["sigma_sq"],
        title="Sigma^2 Prior vs Posterior",
        x_label="sigma^2",
        legend_labels=("Prior", "Posterior"),
    )

    write_svg_density_plot(
        os.path.join(output_dir, "density_mu.svg"),
        *density_data["mu"],
        title="Mu Prior vs Posterior",
        x_label="mu",
        legend_labels=("Prior", "Posterior"),
    )

    write_svg_density_plot(
        os.path.join(output_dir, "density_tau_sq.svg"),
        *density_data["tau_sq"],
        title="Tau^2 Prior vs Posterior",
        x_label="tau^2",
        legend_labels=("Prior", "Posterior"),
    )

    r_posterior = compute_R_samples(samples["sigma_sq"], samples["tau_sq"])
    prior_r = sample_prior_R(20000, 2.0, 10.0, 2.0, 15.0)
    min_r = min(min(r_posterior), min(prior_r))
    max_r = max(max(r_posterior), max(prior_r))
    grid_r = make_grid(min_r, max_r, 200)
    posterior_r_density = kernel_density_estimate(r_posterior, grid_r)
    prior_r_density = kernel_density_estimate(prior_r, grid_r)
    write_svg_density_plot(
        os.path.join(output_dir, "density_R.svg"),
        grid_r,
        posterior_r_density,
        prior_r_density,
        title="R Prior vs Posterior",
        x_label="R",
        legend_labels=("Prior", "Posterior"),
    )

    theta_means = posterior_expectations(samples["theta"])
    group_means = [sample_mean(group) for group in groups]
    write_svg_scatter_plot(
        os.path.join(output_dir, "sample_vs_posterior_theta.svg"),
        group_means,
        theta_means,
        "Sample Means vs Posterior Means",
        "Sample Mean",
        "Posterior Mean",
    )

    all_values = [val for group in groups for val in group]
    sample_global_mean = sample_mean(all_values)

    prob_theta7_lt_theta6 = sum(
        1 for draw in samples["theta"] if draw[6] < draw[5]
    ) / len(samples["theta"])
    prob_theta7_smallest = sum(
        1 for draw in samples["theta"] if draw[6] == min(draw)
    ) / len(samples["theta"])

    write_summary_text(os.path.join(output_dir, "posterior_summary.csv"), summary, ess_values)
    write_additional_metrics(
        os.path.join(output_dir, "additional_metrics.csv"),
        prob_theta7_lt_theta6,
        prob_theta7_smallest,
        sample_global_mean,
        summary["mu"]["mean"],
    )

    return {
        "samples": samples,
        "summary": summary,
        "ess": ess_values,
        "r_posterior": r_posterior,
        "r_prior": prior_r,
        "theta_means": theta_means,
        "group_means": group_means,
        "sample_global_mean": sample_global_mean,
        "prob_theta7_lt_theta6": prob_theta7_lt_theta6,
        "prob_theta7_smallest": prob_theta7_smallest,
    }


if __name__ == "__main__":
    output = summarize_mcmc("data", "figures")
    print("Posterior summary (mean, 95% CI, ESS):")
    for key, info in output["summary"].items():
        print(
            f"{key}: mean={info['mean']:.4f}, 95% CI=({info['lower']:.4f}, {info['upper']:.4f}), ESS={info['ess']:.1f}"
        )
    print(
        f"P(theta7 < theta6) = {output['prob_theta7_lt_theta6']:.4f}, P(theta7 is smallest) = {output['prob_theta7_smallest']:.4f}"
    )
    print(
        f"Sample mean of all observations = {output['sample_global_mean']:.4f}, Posterior mean of mu = {output['summary']['mu']['mean']:.4f}"
    )
