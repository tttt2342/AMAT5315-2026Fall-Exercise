"""Independent spin enumeration and reconstruction from saved tensor evidence."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def read_tensor(directory, entry):
    return np.fromfile(directory / entry["file"], dtype="<f8").reshape(entry["shape"])


def run_solver(binary, method, l, k, chi, out, full=False):
    command = [str(binary.resolve()), "--method", method, "--l", str(l),
               "--k", f"{k:.10g}", "--chi", str(chi), "--out", str(out)]
    if not full:
        command.append("--summary-only")
    env = os.environ.copy()
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, env=env)
    return read_json(out / "run.json")


def replay(directory):
    run = read_json(directory / "run.json")
    assert run["full_evidence"]
    if run["method"] == "boundary":
        details = read_json(directory / "boundary.json")
        assert len(details["cores"]) == len(details["row_scales"]) == run["l"]
        vector = np.ones(1)
        for i, core in enumerate(details["cores"]):
            tensor = read_tensor(directory, core)
            assert max(tensor.shape) <= max(2, run["chi"])
            if i < run["l"] - 1:
                matrix = tensor.reshape(-1, tensor.shape[-1])
                np.testing.assert_allclose(matrix.T @ matrix, np.eye(matrix.shape[1]),
                                           atol=1e-12, rtol=0)
            else:
                assert math.isclose(np.linalg.norm(tensor), 1, abs_tol=1e-12)
            vector = np.einsum("a,apb,p->b", vector, tensor, details["edge_vector"])
        assert len(vector) == 1
        closure = vector[0]
        log_scale = math.fsum(map(math.log, details["row_scales"]))
    elif run["method"] == "trg":
        details = read_json(directory / "trg.json")
        steps = details["steps"]
        assert len(steps) == 2 * (run["l"].bit_length() - 1)
        for i, step in enumerate(steps):
            assert step["sites_before_split"] == run["sites"] >> i
            for cut in ("lu_rd", "dl_ur"):
                split = step[cut]
                spectrum = read_tensor(directory, split["singular_values"])
                assert np.all(spectrum[:-1] >= spectrum[1:]) and np.all(spectrum >= 0)
                rank = split["left"]["shape"][-1]
                assert rank == split["right"]["shape"][0] == min(run["chi"], len(spectrum))
        tensor = read_tensor(directory, details["final"])
        first, second = steps[-1]["lu_rd"], steps[-1]["dl_ur"]
        s1, s3 = (read_tensor(directory, first[side]) for side in ("left", "right"))
        s2, s4 = (read_tensor(directory, second[side]) for side in ("left", "right"))
        regrouped = np.einsum("war,abu,lbg,dgw->ruld", s1, s2, s3, s4, optimize=True)
        np.testing.assert_allclose(regrouped, tensor, atol=1e-12, rtol=1e-10)
        closure = np.einsum("ijij->", tensor)
        log_scale = math.fsum(step["sites_before_split"] * math.log(step["scale"]) for step in steps)
    else:
        raise ValueError(run["method"])

    assert math.isclose(closure, details["closure"], rel_tol=1e-10)
    actual = (log_scale + math.log(closure)) / run["sites"]
    assert math.isclose(actual, run["g"], abs_tol=1e-12, rel_tol=0)
    return actual


def enumerate_spins(l, k, periodic):
    sites = l * l
    states = np.arange(1 << sites, dtype=np.int64)
    energies = np.zeros_like(states)
    for row in range(l):
        for col in range(l):
            site = row * l + col
            if periodic or col + 1 < l:
                right = row * l + (col + 1) % l
                energies += 2 * (((states >> site) ^ (states >> right)) & 1) - 1
            if periodic or row + 1 < l:
                down = ((row + 1) % l) * l + col
                energies += 2 * (((states >> site) ^ (states >> down)) & 1) - 1
    weights = np.exp(-k * energies)
    z = weights.sum()
    p = weights / z
    mean_energy = np.dot(p, energies)
    variance = np.dot(p, (energies - mean_energy) ** 2)
    return math.log(z) / sites, float(k * k * variance / sites)


def curvature(k, h, minus, center, plus):
    return k * k * ((plus - center) - (center - minus)) / h**2


def small_checks(binary, method):
    if method == "boundary":
        cases = [(4, k, 8, 1e-8, k == 0.44) for k in (0., 0.44, 0.7)]
    else:
        cases = [(2, 0.44, 16, 1e-8, True), (4, 0.44, 8, 2e-3, False)]
    checks = []
    with tempfile.TemporaryDirectory(prefix="thermo-checks-") as tmp:
        out = Path(tmp)
        for l, k, chi, tolerance, check_heat in cases:
            run = run_solver(binary, method, l, k, chi, out, full=True)
            replay(out)
            g, cv = enumerate_spins(l, k, method == "trg")
            result = {
                "l": l, "k": k, "chi": chi, "g": run["g"], "exact_g": g,
                "absolute_g_error": abs(g - run["g"]), "g_tolerance": tolerance,
            }
            assert result["absolute_g_error"] < tolerance, result
            if check_heat:
                h = 0.0025
                minus = run_solver(binary, method, l, k - h, chi, out)
                plus = run_solver(binary, method, l, k + h, chi, out)
                estimate = curvature(k, h, minus["g"], run["g"], plus["g"])
                result.update(cv=estimate, exact_cv=cv, absolute_cv_error=abs(estimate - cv))
                assert result["absolute_cv_error"] < 1e-3, result
            checks.append(result)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["boundary", "trg"])
    parser.add_argument("--binary", type=Path, default=Path(".tool/bin/thermo"))
    parser.add_argument("--replay", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    checks = small_checks(args.binary, args.method)
    replays = {str(directory): replay(directory) for directory in args.replay}
    result = {"spin_enumeration": checks, "reconstructed_g": replays}
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    report = json.dumps(result, indent=2) + "\n"
    (out / f"{args.method}-checks.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
