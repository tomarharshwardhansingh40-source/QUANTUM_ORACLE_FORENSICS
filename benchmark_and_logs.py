"""
benchmark_and_logs.py
Generates all Section 9 required baselines, confusion matrices, and budget scaling logs.
"""

from collections import Counter
import numpy as np
from solver import (
    solve,
    make_oracle_table,
    LocalOracleRunner,
    _build_classical_query_circuit,
    _build_bv_circuit,
    _parse_counts_to_ints,
    _bitwise_majority,
    _gf2_dot,
    _hamming_weight,
)

FAMILIES = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]


# --- Baseline 1: Random Guessing ---
def solve_random(n, m, budget, families, run, rng):
    valid = ["constant", "balanced", "linear", "grover", "randfun"] if m == 1 else ["simon", "simon2", "perm"]
    fam = str(rng.choice(valid))
    if fam == "constant":
        return fam, int(rng.integers(0, 2))
    if fam == "linear":
        return fam, [int(rng.integers(1, 256)), int(rng.integers(0, 2))]
    if fam == "grover":
        return fam, int(rng.integers(0, 256))
    if fam == "simon":
        return fam, int(rng.integers(1, 256))
    if fam == "simon2":
        s1, s2 = int(rng.integers(1, 256)), int(rng.integers(1, 256))
        while s2 == s1:
            s2 = int(rng.integers(1, 256))
        return fam, sorted([s1, s2, s1 ^ s2])
    return fam, None


# --- Baseline 2: Classical-Only Strategy (No Superposition) ---
def solve_classical_only(n, m, budget, families, run):
    queries = 0
    outputs = {}
    for x in range(min(budget, 256)):
        qc = _build_classical_query_circuit(n, m, x)
        res = _parse_counts_to_ints(run(qc, 1), m)
        outputs[x] = res[0]
        queries += 1

    if m == 1:
        ones = sum(outputs.values())
        if ones == 0:
            return ("constant", 0)
        if ones == len(outputs):
            return ("constant", 1)
        if ones == 1:
            for x, val in outputs.items():
                if val == 1:
                    return ("grover", x)
        # Check if any linear (a, b) fits all observed points
        for a in range(1, 256):
            b = outputs[0]
            if sum(1 for x, val in outputs.items() if (_gf2_dot(a, x) ^ b) == val) >= len(outputs) - 2:
                return ("linear", [a, b])
        return ("balanced", None) if (ones / len(outputs)) >= 0.4 else ("randfun", None)
    else:
        # Look for exact collisions in classical table
        seen = {}
        found_s = set()
        for x, y in outputs.items():
            if y in seen:
                found_s.add(x ^ seen[y])
            else:
                seen[y] = x
        found_s.discard(0)
        if len(found_s) >= 2:
            s_list = sorted(list(found_s))
            s1, s2 = s_list[0], s_list[1]
            return ("simon2", sorted([s1, s2, s1 ^ s2]))
        elif len(found_s) == 1:
            return ("simon", list(found_s)[0])
        return ("perm", None)


# --- Baseline 3: Single-Algorithm Strategy (Always runs BV circuit) ---
def solve_single_algo_bv(n, m, budget, families, run):
    qc = _build_bv_circuit(n, 1) if m == 1 else _build_bv_circuit(n, 1)
    # Place 1-bit BV on first n+1 qubits regardless of m
    from qiskit import QuantumCircuit
    from solver import _make_oracle_gate
    qc = QuantumCircuit(n + m, n)
    qc.x(n)
    qc.h(n)
    for k in range(n):
        qc.h(k)
    qc.append(_make_oracle_gate(n, m), list(range(n + m)))
    for k in range(n):
        qc.h(k)
        qc.measure(k, k)

    samples = _parse_counts_to_ints(run(qc, budget), n)
    near_zero = sum(1 for s in samples if _hamming_weight(s) <= 1)
    if m == 1:
        if near_zero >= int(0.75 * budget):
            return ("constant", 0)
        maj = _bitwise_majority([s for s in samples if s != 0], n)
        if sum(1 for s in samples if _hamming_weight(s ^ maj) <= 1) >= int(0.5 * budget):
            return ("linear", [maj, 0])
        return ("balanced", None) if near_zero < int(0.1 * budget) else ("randfun", None)
    else:
        return ("perm", None)


def evaluate_all():
    rng = np.random.default_rng(42)
    trials_per_fam = 12  # 96 total oracles (matches Stage 2 Suite S size)

    print("=================================================================")
    print(" QRYPTONOX QUANTUM ORACLE FORENSICS - FULL DIAGNOSTIC LOGS")
    print("=================================================================\n")

    # 1. Confusion Matrix & Standard + Stress Evaluation for Main Solver
    for label, (pm, pf) in [("STANDARD NOISE (4% misfire, 2% flip)", (0.04, 0.02)),
                            ("STRESS NOISE (8% misfire, 4% flip)", (0.08, 0.04))]:
        conf_matrix = {f: Counter() for f in FAMILIES}
        secret_ok = Counter()
        savings = []

        for fam in FAMILIES:
            for _ in range(trials_per_fam):
                table, m, secret = make_oracle_table(fam, rng)
                runner = LocalOracleRunner(table, 8, m, p_misfire=pm, p_flip=pf, seed=int(rng.integers(1e9)))
                pred_fam, pred_sec = solve(8, m, 60, FAMILIES, runner.run)
                conf_matrix[fam][pred_fam] += 1
                if pred_fam == fam and pred_sec == secret:
                    secret_ok[fam] += 1
                    savings.append(1.0 - runner.queries_used / 60.0)

        total_tasks = trials_per_fam * len(FAMILIES)
        overall_acc = sum(secret_ok.values()) / total_tasks
        worst_acc = min(secret_ok[f] / trials_per_fam for f in FAMILIES)
        mean_save = np.mean(savings) if savings else 0.0

        print(f"--- {label} ---")
        print(f"Overall Task Accuracy : {overall_acc*100:.2f}% ({sum(secret_ok.values())}/{total_tasks})")
        print(f"Worst-Family Accuracy : {worst_acc*100:.2f}%")
        print(f"Mean Query Saving     : {mean_save*100:.2f}%\n")

        print("Confusion Matrix (Rows = True Family, Cols = Predicted Family):")
        header = "True \\ Pred | " + " | ".join(f"{f[:6]:6s}" for f in FAMILIES) + " | ExactSecret"
        print(header)
        print("-" * len(header))
        for f in FAMILIES:
            row_str = f"{f:11s} | " + " | ".join(f"{conf_matrix[f][pf]:6d}" for pf in FAMILIES)
            row_str += f" | {secret_ok[f]}/{trials_per_fam}"
            print(row_str)
        print("\n")

    # 2. Required Baselines Comparison
    print("--- SECTION 9 BASELINE COMPARISONS (Budget = 60, Standard Noise) ---")
    for b_name, b_fn in [
        ("Random Guessing", lambda n, m, b, f, r: solve_random(n, m, b, f, r, rng)),
        ("Classical-Only (No Superposition)", solve_classical_only),
        ("Single-Algorithm (BV Only)", solve_single_algo_bv),
        ("Adaptive Hybrid Solver (Ours)", solve),
    ]:
        ok = 0
        saves = []
        for fam in FAMILIES:
            for _ in range(trials_per_fam):
                table, m, secret = make_oracle_table(fam, rng)
                runner = LocalOracleRunner(table, 8, m, 0.04, 0.02, seed=int(rng.integers(1e9)))
                pf, ps = b_fn(8, m, 60, FAMILIES, runner.run)
                if pf == fam and ps == secret:
                    ok += 1
                    saves.append(1.0 - min(runner.queries_used, 60) / 60.0)
        print(f"{b_name:35s} -> Accuracy: {ok/(trials_per_fam*8)*100:5.1f}%, Mean Saving: {(np.mean(saves)*100 if saves else 0):5.1f}%")
    print()

    # 3. Accuracy vs Budget Cap (20, 30, 40, 50, 60)
    print("--- SECTION 9 ACCURACY AS A FUNCTION OF BUDGET CAP ---")
    for b_cap in [20, 30, 40, 50, 60]:
        ok = 0
        for fam in FAMILIES:
            for _ in range(trials_per_fam):
                table, m, secret = make_oracle_table(fam, rng)
                runner = LocalOracleRunner(table, 8, m, 0.04, 0.02, seed=int(rng.integers(1e9)))
                pf, ps = solve(8, m, b_cap, FAMILIES, runner.run)
                if pf == fam and ps == secret and runner.queries_used <= b_cap:
                    ok += 1
        print(f"Budget Cap = {b_cap:2d} queries -> Task Accuracy: {ok/(trials_per_fam*8)*100:5.1f}%")


if __name__ == "__main__":
    evaluate_all()