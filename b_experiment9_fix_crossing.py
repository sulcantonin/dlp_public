"""
DLP Hardware Validation — Crossing Experiment
==============================================
Autonomous adaptive routing under gradually shifting noise.
Outputs all data needed for paper figures and reproducibility audit.

Usage:
  IBM_API_KEY=your_key python b_experiment9_fix_crossing.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os, time, json, traceback
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps as qasm2_dumps

# ---- Publication figure style ----
COLW = 3.5          # single-column width (inches)
DBLW = 7.0          # double-column width (inches)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

OUT = 'figures/b_experiment9_fix_crossing/'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}circuits/', exist_ok=True)


# ---- API key ----
def load_api_key():
    key = os.environ.get('IBM_API_KEY')
    if key: return key
    for p in ['.env', '../.env', os.path.expanduser('~/.env')]:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    for line in f:
                        if line.strip().startswith('IBM_API_KEY='):
                            return line.strip().split('=',1)[1].strip().strip('"').strip("'")
            except: pass
    return None


# ---- Runners ----
class IBMHardwareRunner:
    def __init__(self, api_key, backend_name='ibm_fez'):
        from qiskit_ibm_runtime import QiskitRuntimeService
        print("Connecting to IBM Quantum...")
        self.service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=api_key)
        self.backend = self.service.backend(backend_name)
        status = self.backend.status()
        print(f"Backend: {self.backend.name} ({self.backend.num_qubits} qubits)")
        print(f"Status:  {'operational' if status.operational else 'NOT operational'}")
        print(f"Queue:   {status.pending_jobs} pending jobs")
        if not status.operational:
            raise RuntimeError(f"{backend_name} not operational")
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        self.pm = generate_preset_pass_manager(
            optimization_level=1, target=self.backend.target)
        self.name = self.backend.name

    def run(self, qc, shots=512):
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        transpiled = self.pm.run(qc)
        sampler = Sampler(self.backend)
        job = sampler.run([transpiled], shots=shots)
        print(f"    Job {job.job_id()[:12]}...", end='', flush=True)
        result = job.result()
        print(" done")
        pub = result[0]
        data = pub.data
        for attr in ['meas', 'c']:
            if hasattr(data, attr):
                return getattr(data, attr).get_counts(), transpiled
        return getattr(data, list(vars(data).keys())[0]).get_counts(), transpiled

    def transpile_qc(self, qc):
        return self.pm.run(qc)


class AerRunner:
    def __init__(self, baseline_cx_err=0.01):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(baseline_cx_err, 2), ['cx'])
        nm.add_all_qubit_quantum_error(depolarizing_error(baseline_cx_err/5, 1), ['rx','ry'])
        self.backend = AerSimulator(noise_model=nm)
        self.name = 'aer_simulator'

    def run(self, qc, shots=2048):
        tc = transpile(qc, self.backend)
        counts = self.backend.run(tc, shots=shots).result().get_counts()
        return counts, tc

    def transpile_qc(self, qc):
        return transpile(qc, self.backend)


# ---- Circuits ----
def build_ghz(theta, path='A', noise_angle=0.0):
    qc = QuantumCircuit(3, 3)
    qc.ry(theta, 0)
    if path == 'A': qc.cx(0, 1); qc.cx(1, 2)
    else:           qc.cx(0, 2); qc.cx(2, 1)
    if abs(noise_angle) > 1e-6:
        qc.rx(noise_angle, 0); qc.rx(noise_angle, 1); qc.rx(noise_angle, 2)
    qc.measure([0,1,2], [0,1,2])
    return qc

def ghz_fid(counts, shots):
    return (counts.get('000',0) + counts.get('111',0)) / shots


# ---- DLP Softmax Router (Sec 4.6.3) ----
class DLPSoftmaxRouter(nn.Module):
    def __init__(self, n_paths=2, init_logit=0.0, max_logit=2.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_paths,), init_logit))
        self.n_paths = n_paths
        self.max_logit = max_logit

    def probs(self):
        return F.softmax(self.logits, dim=0)

    def selected_path(self):
        return torch.argmax(self.logits).item()

    def clamp_logits(self):
        with torch.no_grad():
            self.logits.clamp_(-self.max_logit, self.max_logit)

    def loss(self, fidelities):
        p = self.probs()
        fid = torch.tensor(fidelities, dtype=torch.float32)
        return 1.0 - (p * fid).sum()


# ---- Environment ----
class Environment:
    def __init__(self, runner, noise_A, noise_B, num_cycles, shots=512):
        self.runner = runner
        self.noise_A = noise_A
        self.noise_B = noise_B
        self.num_cycles = num_cycles
        self.shots = shots
        self.theta = math.pi / 2

    def run_path(self, path_idx, cycle, shots=None):
        shots = shots or self.shots
        path = 'A' if path_idx == 0 else 'B'
        noise = self.noise_A[cycle] if path_idx == 0 else self.noise_B[cycle]
        qc = build_ghz(self.theta, path=path, noise_angle=noise)
        counts, tqc = self.runner.run(qc, shots)
        return ghz_fid(counts, shots), counts, qc, tqc


# ---- Noise profile ----
def crossing_8cyc(seed=42):
    rng = np.random.RandomState(seed)
    noise_A = np.array([0.03, 0.05, 0.08, 0.4, 0.7, 0.95, 1.1, 1.2]) + rng.normal(0, 0.02, 8)
    noise_B = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.04, 0.03]) + rng.normal(0, 0.02, 8)
    return np.clip(noise_A, 0, 1.5), np.clip(noise_B, 0, 1.2)


# ---- Experiment runner ----
def run_experiment(runner, noise_A, noise_B, num_cycles,
                   shots=512, train_iters=10, lr=0.5):
    torch.manual_seed(42)
    router = DLPSoftmaxRouter(n_paths=2, init_logit=0.0)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    env = Environment(runner, noise_A, noise_B, num_cycles, shots)

    log = {k: [] for k in ['static_fid','rr_fid','greedy_fid','dlp_fid',
           'dlp_path','pA','pB','fid_A','fid_B','loss',
           'logit_A','logit_B','counts_A','counts_B',
           'counts_static','counts_rr']}
    log.update({'backend': runner.name, 'shots': shots,
        'num_cycles': num_cycles, 'noise_A': noise_A.tolist(),
        'noise_B': noise_B.tolist(), 'timestamp': datetime.now().isoformat(),
        'lr': lr, 'train_iters': train_iters, 'init_bias': 0.0})

    # Save circuits (QASM) for both paths at first and last cycle
    circuits_saved = {}

    print(f"\n{'Cyc':>3} {'Sel':>3} {'fid_A':>7} {'fid_B':>7} "
          f"{'DLP':>7} {'Static':>7} {'Greedy':>7} "
          f"{'p(A)':>6} {'p(B)':>6} {'Loss':>8} {'λ_A':>6} {'λ_B':>6}")
    print("-" * 90)
    t0 = time.time()
    total_dlp_shots = 0

    for t in range(num_cycles):
        # ---- Measure BOTH paths ----
        fid_A, counts_A, qc_a, tqc_a = env.run_path(0, t)
        fid_B, counts_B, qc_b, tqc_b = env.run_path(1, t)
        total_dlp_shots += 2 * shots

        # Save QASM + circuit diagrams for selected cycles
        if t in [0, num_cycles // 2, num_cycles - 1]:
            for label, qc, tqc in [('A', qc_a, tqc_a), ('B', qc_b, tqc_b)]:
                prefix = f'cycle{t}_path{label}'
                # QASM
                qasm_str = qasm2_dumps(qc)
                with open(f'{OUT}circuits/{prefix}.qasm', 'w') as f:
                    f.write(qasm_str)
                # Logical circuit diagram
                try:
                    qc.draw('mpl', filename=f'{OUT}circuits/{prefix}.pdf')
                    qc.draw('mpl', filename=f'{OUT}circuits/{prefix}.png')
                    plt.close('all')
                except Exception as e:
                    print(f"    [warn] Could not draw {prefix}: {e}")
                # Transpiled version (the actual circuit that ran on hardware)
                tqasm = qasm2_dumps(tqc)
                with open(f'{OUT}circuits/{prefix}_transpiled.qasm', 'w') as f:
                    f.write(tqasm)
                try:
                    tqc.draw('mpl', filename=f'{OUT}circuits/{prefix}_transpiled.pdf')
                    tqc.draw('mpl', filename=f'{OUT}circuits/{prefix}_transpiled.png')
                    plt.close('all')
                except Exception as e:
                    print(f"    [warn] Could not draw {prefix}_transpiled: {e}")
                circuits_saved[f'{prefix}'] = f'{prefix}.qasm'
                circuits_saved[f'{prefix}_transpiled'] = f'{prefix}_transpiled.qasm'

        # ---- Gradient updates ----
        for _ in range(train_iters):
            opt.zero_grad()
            loss = router.loss([fid_A, fid_B])
            loss.backward()
            opt.step()
            router.clamp_logits()

        # ---- Read state ----
        with torch.no_grad():
            p = router.probs()
            pA, pB = p[0].item(), p[1].item()
            sel = router.selected_path()
            l_val = router.loss([fid_A, fid_B]).item()
            lA = router.logits[0].item()
            lB = router.logits[1].item()

        dlp_fid = fid_A if sel == 0 else fid_B

        # ---- Baselines ----
        fid_static, counts_static, _, _ = env.run_path(0, t)
        fid_rr, counts_rr, _, _ = env.run_path(t % 2, t)
        fid_g0, _, _, _ = env.run_path(0, t)
        fid_g1, _, _, _ = env.run_path(1, t)
        fid_greedy = max(fid_g0, fid_g1)

        # ---- Log everything ----
        log['static_fid'].append(fid_static)
        log['rr_fid'].append(fid_rr)
        log['greedy_fid'].append(fid_greedy)
        log['dlp_fid'].append(dlp_fid)
        log['dlp_path'].append(sel)
        log['pA'].append(pA)
        log['pB'].append(pB)
        log['fid_A'].append(fid_A)
        log['fid_B'].append(fid_B)
        log['loss'].append(l_val)
        log['logit_A'].append(lA)
        log['logit_B'].append(lB)
        log['counts_A'].append(counts_A)
        log['counts_B'].append(counts_B)
        log['counts_static'].append(counts_static)
        log['counts_rr'].append(counts_rr)

        path_label = 'A' if sel == 0 else 'B'
        print(f"{t:3d}   {path_label}   {fid_A:7.3f} {fid_B:7.3f}  "
              f"{dlp_fid:7.3f} {fid_static:7.3f} {fid_greedy:7.3f} "
              f"{pA:6.3f} {pB:6.3f} {l_val:8.4f} {lA:6.3f} {lB:6.3f}")

    log['elapsed_sec'] = time.time() - t0
    log['total_dlp_shots'] = total_dlp_shots
    log['circuits_saved'] = circuits_saved
    return log


# ============================================================
#  Publication-quality figures (separate files)
# ============================================================

def plot_fidelity(log, filename):
    """Fig: Fidelity comparison across calibration cycles."""
    N = log['num_cycles']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 2.2))

    ax.plot(cycles, log['fid_A'], ':', color='#4393C3', lw=0.8, alpha=0.5, label='Path A (meas.)')
    ax.plot(cycles, log['fid_B'], ':', color='#F4A582', lw=0.8, alpha=0.5, label='Path B (meas.)')
    ax.plot(cycles, log['static_fid'], '--', color='#D6604D', lw=1.0, alpha=0.8, label='Static (always A)')
    ax.plot(cycles, log['greedy_fid'], '-', color='0.55', lw=0.8, alpha=0.6, label='Greedy oracle')
    ax.plot(cycles, log['dlp_fid'], '-s', color='#2CA02C', lw=1.5, ms=4, zorder=5, label='DLP Router')

    ax.set_ylabel('GHZ Fidelity')
    ax.set_xlabel('Calibration Cycle')
    ax.set_ylim([0, 1.08])
    ax.set_xlim([-0.3, N - 0.7])
    ax.set_xticks(cycles)
    ax.legend(loc='lower left', fontsize=6.5, ncol=2, framealpha=0.9,
              handlelength=1.8, columnspacing=1.0)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_probabilities(log, filename):
    """Fig: Softmax path probabilities."""
    N = log['num_cycles']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.5))

    ax.plot(cycles, log['pA'], '-o', color='#4393C3', lw=1.3, ms=3.5, label='$p_A$')
    ax.plot(cycles, log['pB'], '-o', color='#F4A582', lw=1.3, ms=3.5, label='$p_B$')
    ax.axhline(0.5, color='grey', ls=':', alpha=0.3, lw=0.5)

    ax.set_ylabel('Path Probability $p_i$')
    ax.set_xlabel('Calibration Cycle')
    ax.set_ylim([-0.05, 1.08])
    ax.set_xlim([-0.3, N - 0.7])
    ax.set_xticks(cycles)
    ax.legend(loc='center right', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_noise_profile(nA, nB, filename):
    """Fig: Injected noise schedule."""
    N = len(nA)
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.3))

    ax.plot(cycles, nA, '-', color='#4393C3', lw=1.2, label='Path A noise')
    ax.plot(cycles, nB, '-', color='#F4A582', lw=1.2, label='Path B noise')

    ax.set_ylabel('$R_x$ noise (rad)')
    ax.set_xlabel('Cycle')
    ax.set_xlim([-0.3, N - 0.7])
    ax.set_xticks(cycles)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_logits(log, filename):
    """Fig: Raw structural logits over time."""
    N = log['num_cycles']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.5))

    ax.plot(cycles, log['logit_A'], '-o', color='#4393C3', lw=1.3, ms=3.5, label='$\\lambda_A$')
    ax.plot(cycles, log['logit_B'], '-o', color='#F4A582', lw=1.3, ms=3.5, label='$\\lambda_B$')
    ax.axhline(0, color='grey', ls=':', alpha=0.3, lw=0.5)

    ax.set_ylabel('Structural Logit $\\lambda_i$')
    ax.set_xlabel('Calibration Cycle')
    ax.set_xlim([-0.3, N - 0.7])
    ax.set_xticks(cycles)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_loss(log, filename):
    """Fig: DLP loss over calibration cycles."""
    N = log['num_cycles']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.5))

    ax.plot(cycles, log['loss'], '-s', color='#2CA02C', lw=1.3, ms=3.5)
    ax.set_ylabel('Loss $\\mathcal{L}$')
    ax.set_xlabel('Calibration Cycle')
    ax.set_xlim([-0.3, N - 0.7])
    ax.set_xticks(cycles)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_combined(log, filename):
    """Fig: Combined fidelity + probability (two-panel, double-column)."""
    N = log['num_cycles']
    cycles = np.arange(N)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DBLW, 3.2), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})

    ax1.plot(cycles, log['fid_A'], ':', color='#4393C3', lw=0.8, alpha=0.5, label='Path A (meas.)')
    ax1.plot(cycles, log['fid_B'], ':', color='#F4A582', lw=0.8, alpha=0.5, label='Path B (meas.)')
    ax1.plot(cycles, log['static_fid'], '--', color='#D6604D', lw=1.0, alpha=0.8, label='Static (always A)')
    ax1.plot(cycles, log['greedy_fid'], '-', color='0.55', lw=0.8, alpha=0.6, label='Greedy oracle')
    ax1.plot(cycles, log['dlp_fid'], '-s', color='#2CA02C', lw=1.5, ms=4, zorder=5, label='DLP Router')
    ax1.set_ylabel('GHZ Fidelity')
    ax1.set_ylim([0, 1.08])
    ax1.set_xlim([-0.3, N - 0.7])
    ax1.legend(loc='lower left', fontsize=6.5, ncol=3, framealpha=0.9,
               handlelength=1.8, columnspacing=1.0)
    ax1.grid(True, alpha=0.25)

    ax2.plot(cycles, log['pA'], '-o', color='#4393C3', lw=1.3, ms=3.5, label='$p_A$')
    ax2.plot(cycles, log['pB'], '-o', color='#F4A582', lw=1.3, ms=3.5, label='$p_B$')
    ax2.axhline(0.5, color='grey', ls=':', alpha=0.3, lw=0.5)
    ax2.set_ylabel('$p_i$')
    ax2.set_xlabel('Calibration Cycle')
    ax2.set_ylim([-0.05, 1.08])
    ax2.set_xticks(cycles)
    ax2.legend(loc='center right', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def print_summary(log, label):
    N = log['num_cycles']
    print(f"\n{'='*65}")
    print(f"RESULTS: {label} ({log['backend']})")
    print(f"{'='*65}")
    print(f"  Static (always A): {np.mean(log['static_fid']):.4f}")
    print(f"  Round-robin:       {np.mean(log['rr_fid']):.4f}")
    print(f"  DLP Adaptive:      {np.mean(log['dlp_fid']):.4f}  ({log['total_dlp_shots']} shots)")
    print(f"  Greedy oracle:     {np.mean(log['greedy_fid']):.4f}")
    print(f"  DLP vs static:     {(np.mean(log['dlp_fid'])-np.mean(log['static_fid']))*100:+.1f} pp")
    print(f"  DLP vs greedy:     {(np.mean(log['dlp_fid'])-np.mean(log['greedy_fid']))*100:+.1f} pp")
    print(f"  Final logits:      λ_A={log['logit_A'][-1]:.4f}, λ_B={log['logit_B'][-1]:.4f}")
    print(f"  Final probs:       p(A)={log['pA'][-1]:.4f}, p(B)={log['pB'][-1]:.4f}")
    print(f"  Path choices:      {['A' if p==0 else 'B' for p in log['dlp_path']]}")
    print(f"  Circuits saved:    {len(log.get('circuits_saved',{}))} files in {OUT}circuits/")
    print(f"  Time: {log['elapsed_sec']:.0f}s")
    print(f"{'='*65}")


# ---- Main ----
if __name__ == '__main__':
    t_total = time.time()

    api_key = load_api_key()
    hw = None
    if api_key:
        print(f"API key found ({api_key[:8]}...)")
        try:
            hw = IBMHardwareRunner(api_key, backend_name='ibm_fez')
        except Exception as e:
            print(f"  !! HW connection failed: {e}")
            traceback.print_exc()
    else:
        print("No API key found — Aer simulation only.")

    aer = AerRunner(baseline_cx_err=0.01)
    runner = hw if hw else aer
    tag = 'hw' if hw else 'sim'
    shots = 512 if hw else 2048

    print(f"\n{'='*65}")
    print(f"CROSSING (8 cycles) on {runner.name}")
    print(f"  A: good→bad, B: bad→good, cross ~cycle 4")
    print(f"  DLP Softmax Router: L = 1 - Σ p_i·F_i")
    print(f"{'='*65}")

    nA, nB = crossing_8cyc(seed=42)

    log = run_experiment(runner, nA, nB, num_cycles=8,
                         shots=shots, train_iters=10, lr=0.5)

    # ---- Generate all figures ----
    plot_fidelity(log, f'fidelity_{tag}')
    plot_probabilities(log, f'probabilities_{tag}')
    plot_noise_profile(nA, nB, f'noise_{tag}')
    plot_logits(log, f'logits_{tag}')
    plot_loss(log, f'loss_{tag}')
    plot_combined(log, f'combined_{tag}')

    print_summary(log, f"Crossing ({tag})")

    # ---- Save full results JSON ----
    # Convert counts dicts to be JSON-safe
    save_log = {k: v for k, v in log.items()}
    for k in ['counts_A', 'counts_B', 'counts_static', 'counts_rr']:
        save_log[k] = [dict(c) for c in save_log[k]]

    with open(f'{OUT}results_{tag}.json', 'w') as f:
        json.dump(save_log, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print(f"\nDONE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"All outputs in: {OUT}")