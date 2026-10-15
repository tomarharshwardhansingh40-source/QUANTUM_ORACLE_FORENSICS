"""
solver.py - Quantum Oracle Forensics (QryptoNox x QC@IITI Fall Fest 2026)
Exact Bayesian Log-Likelihood GF(2) Decoder + 6-Iteration Grover Peak Concentrator.
"""

from collections import Counter
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


# ============================================================================
# HELPER FUNCTIONS: CIRCUIT BUILDERS & GF(2) / NOISE UTILS
# ============================================================================

def _make_oracle_gate(n: int, m: int) -> Gate:
    """Creates the mandatory placeholder oracle gate on n + m qubits."""
    return Gate("oracle", n + m, [])


def _parse_counts_to_ints(counts: dict, n_bits: int) -> list[int]:
    """Converts Qiskit counts dictionary into a list of integer outcomes."""
    samples = []
    for bitstr, cnt in counts.items():
        clean_str = bitstr.replace(" ", "")[-n_bits:]
        val = int(clean_str, 2)
        samples.extend([val] * cnt)
    return samples


def _hamming_weight(x: int) -> int:
    return int(x).bit_count()


def _hamming_distance(x: int, y: int) -> int:
    return int(x ^ y).bit_count()


def _gf2_dot(x: int, y: int) -> int:
    return (int(x) & int(y)).bit_count() & 1


def _build_classical_query_circuit(n: int, m: int, x_val: int) -> QuantumCircuit:
    """Prepares |x_val>|0^m>, applies the oracle once, and measures the m output qubits."""
    qc = QuantumCircuit(n + m, m)
    for k in range(n):
        if (x_val >> k) & 1:
            qc.x(k)
    qc.append(_make_oracle_gate(n, m), list(range(n + m)))
    for j in range(m):
        qc.measure(n + j, j)
    return qc


def _build_bv_circuit(n: int, m: int = 1) -> QuantumCircuit:
    """Bernstein-Vazirani / Deutsch-Jozsa circuit for m=1."""
    qc = QuantumCircuit(n + m, n)
    qc.x(n)
    qc.h(n)
    for k in range(n):
        qc.h(k)
    qc.append(_make_oracle_gate(n, m), list(range(n + m)))
    for k in range(n):
        qc.h(k)
        qc.measure(k, k)
    return qc


def _build_grover_circuit(n: int, iterations: int) -> QuantumCircuit:
    """Constructs an r-iteration Grover circuit on n input qubits and m=1 ancilla in |->."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(n)
    for k in range(n):
        qc.h(k)

    oracle_gate = _make_oracle_gate(n, 1)
    input_qubits = list(range(n))
    all_qubits = list(range(n + 1))

    for _ in range(iterations):
        qc.append(oracle_gate, all_qubits)
        for k in input_qubits:
            qc.h(k)
            qc.x(k)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for k in input_qubits:
            qc.x(k)
            qc.h(k)

    for k in range(n):
        qc.measure(k, k)
    return qc


def _build_simon_circuit(n: int, m: int) -> QuantumCircuit:
    """Simon's circuit for m=8: applies H^n -> Oracle -> H^n and measures n input qubits."""
    qc = QuantumCircuit(n + m, n)
    for k in range(n):
        qc.h(k)
    qc.append(_make_oracle_gate(n, m), list(range(n + m)))
    for k in range(n):
        qc.h(k)
        qc.measure(k, k)
    return qc


def _bitwise_majority(samples: list[int], n_bits: int) -> int:
    """Computes the bitwise majority vote across a list of integer samples."""
    if not samples:
        return 0
    total = len(samples)
    res = 0
    for k in range(n_bits):
        ones = sum((s >> k) & 1 for s in samples)
        if ones * 2 >= total:
            res |= (1 << k)
    return res


# ============================================================================
# MAIN SOLVER ENTRY POINT
# ============================================================================

def solve(n: int, m: int, budget: int, families: list[str], run) -> tuple[str, any]:
    """
    Identifies the sealed oracle family and recovers its secret within `budget` queries.
    """
    queries_used = 0

    def safe_run(qc: QuantumCircuit, shots: int, gates_per_shot: int = 1) -> list[int]:
        nonlocal queries_used
        cost = shots * gates_per_shot
        if queries_used + cost > budget:
            shots = (budget - queries_used) // gates_per_shot
            if shots <= 0:
                return []
            cost = shots * gates_per_shot
        queries_used += cost
        counts = run(qc, shots)
        return _parse_counts_to_ints(counts, qc.num_clbits)

    def query_classical_raw(x_val: int, shots: int = 1) -> list[int]:
        qc = _build_classical_query_circuit(n, m, x_val)
        return safe_run(qc, shots, gates_per_shot=1)

    def query_classical(x_val: int, shots: int = 1) -> int:
        samples = query_classical_raw(x_val, shots)
        return _bitwise_majority(samples, m)

    # ========================================================================
    # CASE 1: m == 1 (constant, balanced, linear, grover, randfun)
    # ========================================================================
    if m == 1:
        # Phase 1: Run 7 Bernstein-Vazirani shots
        bv_shots = min(7, budget)
        bv_samples = safe_run(_build_bv_circuit(n, 1), bv_shots, gates_per_shot=1)

        # In Stress Mode (4% bit-flip), a 0^8 state has weight <= 1 w.p. 96.2% and weight <= 2 w.p. 99.7%
        near_zero_w1 = sum(1 for s in bv_samples if _hamming_weight(s) <= 1)
        near_zero_w2 = sum(1 for s in bv_samples if _hamming_weight(s) <= 2)

        # Look for a non-zero cluster (candidate hidden mask `a` for `linear`)
        non_zero_samples = [s for s in bv_samples if _hamming_weight(s) > 1]
        if not non_zero_samples:
            non_zero_samples = [s for s in bv_samples if s != 0]

        cand_a = _bitwise_majority(non_zero_samples, n) if non_zero_samples else 0
        best_a = 0
        best_cluster = 0
        candidates_to_check = {cand_a} | set(non_zero_samples)
        for c in candidates_to_check:
            if c == 0:
                continue
            cluster_sz = sum(1 for s in bv_samples if _hamming_distance(s, c) <= 1)
            if cluster_sz > best_cluster:
                best_cluster = cluster_sz
                best_a = c

        # --------------------------------------------------------------------
        # Branch 1A: Non-zero BV cluster -> Test `linear`
        # --------------------------------------------------------------------
        if best_a != 0 and best_cluster >= 3 and near_zero_w1 <= 4:
            if best_cluster <= 5 and queries_used + 2 <= budget:
                extra_bv = safe_run(_build_bv_circuit(n, 1), 2, gates_per_shot=1)
                bv_samples.extend(extra_bv)
                near_zero_w1 = sum(1 for s in bv_samples if _hamming_weight(s) <= 1)

            inliers = [s for s in bv_samples if _hamming_distance(s, best_a) <= 2 and s != 0]
            refined_a = _bitwise_majority(inliers, n) if inliers else best_a
            if refined_a == 0:
                refined_a = best_a

            b0 = query_classical(0, shots=3)
            lsb_bit = (refined_a & -refined_a).bit_length() - 1
            x1 = 1 << lsb_bit
            b1 = query_classical(x1, shots=3)

            if b1 == (b0 ^ 1):
                x2 = 255 if _gf2_dot(refined_a, 255) == 0 else (255 ^ x1)
                b2 = query_classical(x2, shots=3)
                if b2 == b0:
                    return ("linear", [int(refined_a), int(b0)])

        # --------------------------------------------------------------------
        # Branch 1B: Strong 0^n BV peak -> `constant` or `grover`
        # --------------------------------------------------------------------
        if near_zero_w1 >= 4 or near_zero_w2 >= 6:
            # Fast 2-query check for `constant` (c=1)
            c0 = query_classical(0, shots=1)
            c255 = query_classical(255, shots=1)
            if c0 + c255 >= 1:
                c85 = query_classical(85, shots=1)
                if c0 + c255 + c85 >= 2:
                    return ("constant", 1)

            # We have 60 - 9 = 51 queries left to separate `grover` from `constant` (c=0)!
            # Use 6-iteration Grover (ideal success prob = 94.8%!) mixed with 5-iteration Grover (81.3%)
            # 4 shots * 6 iters (24q) + 3 shots * 5 iters (15q) = 39 queries -> leaves 11-12 queries for verification!
            g_samples = []
            if budget - queries_used >= 39:
                g_samples.extend(safe_run(_build_grover_circuit(n, 6), 4, gates_per_shot=6))
                g_samples.extend(safe_run(_build_grover_circuit(n, 5), 3, gates_per_shot=5))
            else:
                while budget - queries_used >= 10:
                    g_samples.extend(safe_run(_build_grover_circuit(n, 5), 1, gates_per_shot=5))

            # Score all 256 possible targets by cluster density around Grover measurements
            cand_scores = Counter()
            for c in range(256):
                exact = sum(1 for s in g_samples if s == c)
                dist1 = sum(1 for s in g_samples if _hamming_distance(s, c) == 1)
                dist2 = sum(1 for s in g_samples if _hamming_distance(s, c) == 2)
                if exact > 0 or dist1 >= 2:
                    cand_scores[c] = exact * 20 + dist1 * 5 + dist2

            # Also add bitwise majority of the tightest 3-shot cluster
            if len(g_samples) >= 3:
                best_center = max(g_samples, key=lambda x: sum(1 for y in g_samples if _hamming_distance(x, y) <= 2))
                cluster_pts = [y for y in g_samples if _hamming_distance(best_center, y) <= 2]
                maj_t = _bitwise_majority(cluster_pts, n)
                cand_scores[maj_t] += 15

            ranked_t = [c for c, _ in cand_scores.most_common(12)]
            for s in g_samples:
                if s not in ranked_t:
                    ranked_t.append(s)

            # Verify top candidates classically
            checked_t = set()
            for t_cand in ranked_t:
                if t_cand in checked_t or queries_used >= budget:
                    continue
                checked_t.add(t_cand)

                raw1 = query_classical_raw(t_cand, shots=1)
                if raw1 and raw1[0] == 1:
                    if queries_used + 2 <= budget:
                        if query_classical(t_cand, shots=2) == 1:
                            return ("grover", int(t_cand))
                    else:
                        return ("grover", int(t_cand))

            # Re-verify top 2 candidates if their 1-shot test suffered an 8% oracle misfire
            for t_cand in ranked_t[:2]:
                if queries_used >= budget:
                    break
                raw_retry = query_classical_raw(t_cand, shots=1)
                if raw_retry and raw_retry[0] == 1:
                    if queries_used + 2 <= budget:
                        if query_classical(t_cand, shots=2) == 1:
                            return ("grover", int(t_cand))
                    else:
                        return ("grover", int(t_cand))

            return ("constant", 0)

        # --------------------------------------------------------------------
        # Branch 1C: Scattered BV spectrum -> `balanced` vs `randfun`
        # --------------------------------------------------------------------
        # Spend remaining budget (up to 48 classical queries) + 5 extra BV shots to measure P_BV(0^n)
        if budget - queries_used >= 45:
            extra_bv = safe_run(_build_bv_circuit(n, 1), 5, gates_per_shot=1)
            bv_samples.extend(extra_bv)
            near_zero_w1 = sum(1 for s in bv_samples if _hamming_weight(s) <= 1)

        num_probes = min(46, budget - queries_used)
        ones_count = 0
        for i in range(num_probes):
            x_val = (i * 73 + 19) & 255
            raw = query_classical_raw(x_val, shots=1)
            if raw:
                ones_count += raw[0]

        emp_rate = ones_count / max(1, num_probes)
        # Safety catch if a `constant` oracle slipped into Branch 1C due to heavy bit-flips
        if emp_rate >= 0.80:
            return ("constant", 1)
        if emp_rate <= 0.08:
            return ("constant", 0)

        # In 12 BV shots:
        # `balanced` has P_ideal(0^n) = 0 -> expected near_zero_w1 is ~0.5
        # `randfun`  has P_ideal(0^n) = 0.16 -> expected near_zero_w1 is ~2.4
        bv_zero_rate = near_zero_w1 / max(1, len(bv_samples))
        # Combined linear discriminant:
        combined_metric = emp_rate - 0.18 * bv_zero_rate
        if combined_metric >= 0.355:
            return ("balanced", None)
        else:
            return ("randfun", None)

    # ========================================================================
    # CASE 2: m == 8 (simon, simon2, perm)
    # ========================================================================
    else:
        # Phase 1: Run 22 shots of Simon's circuit (22 queries)
        simon_shots = min(22, budget)
        z_samples = safe_run(_build_simon_circuit(n, m), simon_shots, gates_per_shot=1)
        non_zero_z = [z for z in z_samples if z != 0]

        # Score every non-zero candidate s in {1..255} using exact Noisy GF(2) Log-Likelihood!
        # Account for the fact that high Hamming weight `s` flips parity (z . s) more often under bit-flip noise!
        p_flip_est = 0.028  # Optimal compromise between 2% standard and 4% stress
        llr_scores = np.zeros(256, dtype=float)
        ortho_counts = np.zeros(256, dtype=int)

        for s in range(1, 256):
            wt = _hamming_weight(s)
            # Probability that parity (z . s) is preserved without flipping:
            parity_keep = 0.94 * ((1.0 - 2.0 * p_flip_est) ** wt)
            p_even = 0.5 * (1.0 + parity_keep)
            p_odd = 0.5 * (1.0 - parity_keep)
            log_even = np.log(p_even / 0.5)
            log_odd = np.log(max(p_odd, 1e-6) / 0.5)

            cnt_even = sum(1 for z in non_zero_z if _gf2_dot(z, s) == 0)
            cnt_odd = len(non_zero_z) - cnt_even
            ortho_counts[s] = cnt_even
            llr_scores[s] = cnt_even * log_even + cnt_odd * log_odd

        ranked_s = sorted(range(1, 256), key=lambda s: llr_scores[s], reverse=True)

        # Phase 2: Two-Point Exact Coset Verification
        y0 = query_classical(0, shots=3)
        y_cache = {0: y0}

        def verify_period(s_cand: int) -> bool:
            """Verifies f(0) == f(s_cand) and f(x_test) == f(x_test ^ s_cand)."""
            if s_cand <= 0 or s_cand >= 256 or queries_used >= budget:
                return False

            if s_cand not in y_cache:
                raw = query_classical_raw(s_cand, shots=1)
                if not raw:
                    return False
                y_s = raw[0]
            else:
                y_s = y_cache[s_cand]

            dist0 = _hamming_distance(y_s, y0)
            if dist0 in (1, 2) and queries_used + 2 <= budget:
                extra = query_classical_raw(s_cand, shots=2)
                y_s = _bitwise_majority([y_s] + extra, m)
                y_cache[s_cand] = y_s
                dist0 = _hamming_distance(y_s, y0)
            else:
                y_cache[s_cand] = y_s

            if dist0 > 1:
                return False

            # Second independent point check x_test vs (x_test ^ s_cand)
            x_test = 1 if s_cand != 1 else 2
            x_partner = x_test ^ s_cand
            if queries_used + 2 <= budget:
                y_a = query_classical(x_test, shots=1)
                y_b = query_classical(x_partner, shots=1)
                if _hamming_distance(y_a, y_b) <= 1:
                    return True
                if _hamming_distance(y_a, y_b) <= 3 and queries_used + 2 <= budget:
                    y_a2 = query_classical(x_test, shots=1)
                    y_b2 = query_classical(x_partner, shots=1)
                    return _hamming_distance(y_a2, y_b2) <= 1
                return False
            else:
                return dist0 == 0

        verified_H = []

        # Screen the top 10 Simon candidates by LLR
        for s_cand in ranked_s[:10]:
            if queries_used >= budget:
                break
            if verify_period(s_cand):
                verified_H.append(s_cand)
                break

        # If #1 or #2 candidate had an oracle misfire on its 1-shot test, re-probe top 3 once
        if not verified_H:
            for s_cand in ranked_s[:3]:
                if queries_used + 2 > budget:
                    break
                y_cache.pop(s_cand, None)
                if verify_period(s_cand):
                    verified_H.append(s_cand)
                    break

        # If we found s1, check if there is a second generator s2 (`simon2` vs `simon`)
        if len(verified_H) == 1:
            s1 = verified_H[0]
            pair_candidates = []
            for s2 in range(1, 256):
                if s2 == s1:
                    continue
                s3 = s1 ^ s2
                if s2 < s3:
                    pair_candidates.append((llr_scores[s2] + llr_scores[s3], s2, s3))
            pair_candidates.sort(reverse=True)

            for _, s2, s3 in pair_candidates[:6]:
                if queries_used >= budget:
                    break
                if verify_period(s2):
                    verified_H.extend([s2, s3])
                    break

            # If s2 misfired on its 1-shot screen, test its coset partner s3 = s1 ^ s2!
            if len(verified_H) == 1:
                for _, s2, s3 in pair_candidates[:2]:
                    if queries_used >= budget:
                        break
                    if verify_period(s3):
                        verified_H.extend([s2, s3])
                        break

        if len(verified_H) >= 2:
            s1, s2 = verified_H[0], verified_H[1]
            subspace = sorted(list({s1, s2, s1 ^ s2}))
            return ("simon2", subspace)
        elif len(verified_H) == 1:
            return ("simon", int(verified_H[0]))
        else:
            return ("perm", None)


# ============================================================================
# LOCAL SIMULATOR, REFERENCE VERIFICATION & BASELINE EVALUATION HARNESS
# ============================================================================

def make_oracle_table(family: str, rng: np.random.Generator, params: dict = None):
    """Generates a truth table f(0..255) and ground-truth secret per Section 3 & 4."""
    params = params or {}
    if family == "constant":
        c = params.get("c", int(rng.integers(0, 2)))
        table = np.full(256, c, dtype=int)
        return table, 1, int(c)

    elif family == "balanced":
        table = np.zeros(256, dtype=int)
        ones_idx = rng.choice(256, size=128, replace=False)
        table[ones_idx] = 1
        return table, 1, None

    elif family == "linear":
        a = params.get("a", int(rng.integers(1, 256)))
        b = params.get("b", int(rng.integers(0, 2)))
        table = np.array([(_gf2_dot(a, x) ^ b) for x in range(256)], dtype=int)
        return table, 1, [int(a), int(b)]

    elif family == "grover":
        t = params.get("t", int(rng.integers(0, 256)))
        table = np.zeros(256, dtype=int)
        table[t] = 1
        return table, 1, int(t)

    elif family == "randfun":
        table = (rng.random(256) < 0.3).astype(int)
        return table, 1, None

    elif family == "simon":
        s = params.get("s", int(rng.integers(1, 256)))
        if params.get("canonical", False):
            table = np.array([min(x, x ^ s) for x in range(256)], dtype=int)
        else:
            vals = rng.permutation(256)
            table = np.zeros(256, dtype=int)
            visited = set()
            idx = 0
            for x in range(256):
                if x not in visited:
                    table[x] = vals[idx]
                    table[x ^ s] = vals[idx]
                    visited.add(x)
                    visited.add(x ^ s)
                    idx += 1
        return table, 8, int(s)

    elif family == "simon2":
        if "H" in params:
            H = params["H"]
            s1, s2 = H[1], H[2]
        else:
            s1 = int(rng.integers(1, 256))
            s2 = int(rng.integers(1, 256))
            while s2 == s1:
                s2 = int(rng.integers(1, 256))
            H = [0, s1, s2, s1 ^ s2]
        if params.get("canonical", False):
            table = np.array([min(x ^ h for h in H) for x in range(256)], dtype=int)
        else:
            vals = rng.permutation(256)
            table = np.zeros(256, dtype=int)
            visited = set()
            idx = 0
            for x in range(256):
                if x not in visited:
                    for h in H:
                        table[x ^ h] = vals[idx]
                        visited.add(x ^ h)
                    idx += 1
        secret = sorted([H[1], H[2], H[3]])
        return table, 8, secret

    elif family == "perm":
        if params.get("canonical", False):
            table = np.array([(37 * x + 11) % 256 for x in range(256)], dtype=int)
        else:
            table = rng.permutation(256).astype(int)
        return table, 8, None

    raise ValueError(f"Unknown family: {family}")


class LocalOracleRunner:
    """
    Fast statevector simulator supporting Qiskit 2.x circuits with Gate("oracle", n+m, []),
    per-gate misfire probability `p_misfire` (acts as identity), and per-bit readout flip `p_flip`.
    """

    def __init__(self, table: np.ndarray, n: int, m: int, p_misfire: float = 0.04, p_flip: float = 0.02, seed: int = None):
        self.table = table
        self.n = n
        self.m = m
        self.p_misfire = p_misfire
        self.p_flip = p_flip
        self.rng = np.random.default_rng(seed)
        self.queries_used = 0

    def run(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        num_q = circuit.num_qubits
        num_c = circuit.num_clbits
        dim = 1 << num_q

        oracle_count = sum(1 for inst in circuit.data if inst.operation.name == "oracle")
        self.queries_used += oracle_count * shots

        ops = []
        meas_map = {}
        for inst in circuit.data:
            name = inst.operation.name
            q_indices = [circuit.find_bit(q).index for q in inst.qubits]
            if name == "measure":
                c_idx = circuit.find_bit(inst.clbits[0]).index
                meas_map[c_idx] = q_indices[0]
            else:
                ops.append((name, q_indices))

        counts = Counter()
        for _ in range(shots):
            misfired_flags = [self.rng.random() < self.p_misfire for _ in range(oracle_count)]
            if not any(misfired_flags) and hasattr(self, "_cached_probs") and self._cached_circuit_id == id(circuit):
                probs = self._cached_probs
            else:
                sv = self._simulate_statevector(num_q, ops, misfired_flags)
                probs = np.abs(sv) ** 2
                probs /= probs.sum()
                if not any(misfired_flags):
                    self._cached_circuit_id = id(circuit)
                    self._cached_probs = probs

            outcome_state = int(self.rng.choice(dim, p=probs))
            bits = []
            for c_idx in range(num_c - 1, -1, -1):
                q_idx = meas_map.get(c_idx, c_idx)
                bit = (outcome_state >> q_idx) & 1
                if self.rng.random() < self.p_flip:
                    bit ^= 1
                bits.append(str(bit))
            counts["".join(bits)] += 1

        return dict(counts)

    def _simulate_statevector(self, num_q: int, ops: list, misfired_flags: list[bool]) -> np.ndarray:
        sv = np.zeros(1 << num_q, dtype=complex)
        sv[0] = 1.0
        sv = sv.reshape([2] * num_q)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        H_mat = np.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=complex)
        X_mat = np.array([[0, 1], [1, 0]], dtype=complex)

        oracle_idx = 0
        for name, q_indices in ops:
            if name == "h":
                q = q_indices[0]
                axis = num_q - 1 - q
                sv = np.tensordot(H_mat, sv, axes=([1], [axis]))
                sv = np.moveaxis(sv, 0, axis)
            elif name == "x":
                q = q_indices[0]
                axis = num_q - 1 - q
                sv = np.tensordot(X_mat, sv, axes=([1], [axis]))
                sv = np.moveaxis(sv, 0, axis)
            elif name == "mcx":
                sv_flat = sv.reshape(-1)
                ctrl_mask = sum(1 << q for q in q_indices[:-1])
                targ_bit = 1 << q_indices[-1]
                idx = np.arange(1 << num_q)
                active = (idx & ctrl_mask) == ctrl_mask
                partner = idx[active] ^ targ_bit
                sv_flat[active] = sv_flat[partner]
                sv = sv_flat.reshape([2] * num_q)
            elif name == "oracle":
                misfire = misfired_flags[oracle_idx]
                oracle_idx += 1
                if not misfire:
                    sv_flat = sv.reshape(-1)
                    idx = np.arange(1 << num_q)
                    x_val = np.zeros(1 << num_q, dtype=int)
                    for k, q in enumerate(q_indices[: self.n]):
                        x_val |= ((idx >> q) & 1) << k
                    f_val = self.table[x_val]
                    new_idx = idx.copy()
                    for j, q in enumerate(q_indices[self.n : self.n + self.m]):
                        new_idx ^= ((f_val >> j) & 1) << q
                    sv_flat = sv_flat[new_idx]
                    sv = sv_flat.reshape([2] * num_q)
        return sv.reshape(-1)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    families = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]

    print("=== Verifying Section 4 Reference Examples ===")
    t_lin, _, _ = make_oracle_table("linear", rng, {"a": 178, "b": 1})
    assert (t_lin[0], t_lin[1], t_lin[2], t_lin[255]) == (1, 1, 0, 1)
    t_sim, _, _ = make_oracle_table("simon", rng, {"s": 105, "canonical": True})
    assert (t_sim[5], t_sim[108], t_sim[105], t_sim[200]) == (5, 5, 0, 161)
    t_sim2, _, _ = make_oracle_table("simon2", rng, {"H": [0, 15, 49, 62], "canonical": True})
    assert (t_sim2[0], t_sim2[15], t_sim2[49], t_sim2[62], t_sim2[1], t_sim2[200]) == (0, 0, 0, 0, 1, 199)
    t_perm, _, _ = make_oracle_table("perm", rng, {"canonical": True})
    assert (t_perm[0], t_perm[1], t_perm[2], t_perm[255]) == (11, 48, 85, 230)
    print("All Section 4 reference checks passed!\n")

    for mode, (pm, pf) in [("Standard (4%/2%)", (0.04, 0.02)), ("Stress (8%/4%)", (0.08, 0.04))]:
        correct_by_fam = Counter()
        total_by_fam = Counter()
        savings = []
        trials_per_fam = 12

        for fam in families:
            for _ in range(trials_per_fam):
                table, m, true_secret = make_oracle_table(fam, rng)
                runner = LocalOracleRunner(table, 8, m, p_misfire=pm, p_flip=pf, seed=int(rng.integers(1e9)))
                pred_fam, pred_secret = solve(8, m, 60, families, runner.run)
                total_by_fam[fam] += 1
                is_ok = (pred_fam == fam) and (pred_secret == true_secret)
                if is_ok:
                    correct_by_fam[fam] += 1
                    savings.append(1.0 - runner.queries_used / 60.0)

        acc = sum(correct_by_fam.values()) / sum(total_by_fam.values())
        worst_acc = min(correct_by_fam[f] / total_by_fam[f] for f in families)
        mean_saving = np.mean(savings) if savings else 0.0
        print(f"=== {mode} Suite (96 Oracles) ===")
        print(f"Overall Accuracy : {acc*100:.1f}%")
        print(f"Worst-Family Acc : {worst_acc*100:.1f}%")
        print(f"Mean Query Saving: {mean_saving*100:.1f}%")
        for f in families:
            print(f"  {f:10s}: {correct_by_fam[f]}/{total_by_fam[f]}")
        print()