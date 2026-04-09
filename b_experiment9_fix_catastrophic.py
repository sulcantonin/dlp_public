"""
DLP Hardware Validation — Catastrophic Failure (Unbiased)
==========================================================
Calibration-based adaptive routing under sudden hardware failure.
Outputs all data needed for paper figures and reproducibility audit.

Usage:
  IBM_API_KEY=your_key python b_experiment9_fix_catastrophic.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os, time, json, traceback
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps as qasm2_dumps

# ---- Publication figure style ----
COLW = 3.5
DBLW = 7.0
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

OUT = 'figures/b_experiment9_fix_catastrophic/'
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


# ---- DLP Softmax Router ----
class DLPSoftmaxRouter(nn.Module):
    def __init__(self, n_paths=2, init_logit=0.0, max_logit=3.0):
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


# ---- Calibration ----
def calibrate_paths(runner, shots=512, n_samples=5):
    theta = math.pi / 2
    fids = {'A': [], 'B': []}

    print(f"\n  === CALIBRATION ({n_samples} samples, {shots} shots each) ===")
    for i in range(n_samples):
        for path_label in ['A', 'B']:
            qc = build_ghz(theta, path=path_label, noise_angle=0.0)
            counts, _ = runner.run(qc, shots)
            f = ghz_fid(counts, shots)
            fids[path_label].append(f)
            print(f"    Cal {i+1}/{n_samples}: Path {path_label} = {f:.4f}")

    avg_A, avg_B = np.mean(fids['A']), np.mean(fids['B'])
    std_A, std_B = np.std(fids['A']), np.std(fids['B'])
    preferred = 0 if avg_A >= avg_B else 1
    pref_label = 'A' if preferred == 0 else 'B'
    other_label = 'B' if preferred == 0 else 'A'

    print(f"\n  Path A: {avg_A:.4f} ± {std_A:.4f}")
    print(f"  Path B: {avg_B:.4f} ± {std_B:.4f}")
    print(f"  Δ = {abs(avg_A - avg_B):.4f}")
    print(f"  → HW prefers Path {pref_label}; failure targets it")
    print(f"  → Survivor: Path {other_label}\n")

    return preferred, fids, {
        'avg_A': avg_A, 'avg_B': avg_B,
        'std_A': std_A, 'std_B': std_B,
        'fids_A': fids['A'], 'fids_B': fids['B'],
    }


# ---- Experiment runner ----
def run_catastrophic(runner, num_cycles=10, fail_cycle=5, fail_noise=1.2,
                     shots=512, train_iters=10, lr=0.5, n_cal_samples=5):

    # ---- Calibrate ----
    preferred, cal_fids, cal_stats = calibrate_paths(
        runner, shots=shots, n_samples=n_cal_samples)
    pref_label = 'A' if preferred == 0 else 'B'
    surv_label = 'B' if preferred == 0 else 'A'

    # ---- Noise arrays ----
    noise_A = np.zeros(num_cycles)
    noise_B = np.zeros(num_cycles)
    for t in range(fail_cycle, num_cycles):
        if preferred == 0:
            noise_A[t] = fail_noise
        else:
            noise_B[t] = fail_noise

    print(f"  Noise: cycles 0–{fail_cycle-1} clean; "
          f"cycles {fail_cycle}–{num_cycles-1}: {fail_noise:.1f} rad on Path {pref_label}\n")

    # ---- Router ----
    torch.manual_seed(42)
    router = DLPSoftmaxRouter(n_paths=2, init_logit=0.0)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    theta = math.pi / 2

    log = {k: [] for k in ['static_fid','rr_fid','greedy_fid','dlp_fid',
                            'dlp_path','pA','pB','fid_A','fid_B','loss',
                            'logit_A','logit_B','counts_A','counts_B',
                            'counts_static','counts_rr']}
    log.update({
        'backend': runner.name, 'shots': shots, 'num_cycles': num_cycles,
        'noise_A': noise_A.tolist(), 'noise_B': noise_B.tolist(),
        'timestamp': datetime.now().isoformat(),
        'lr': lr, 'train_iters': train_iters,
        'preferred_path': preferred, 'preferred_label': pref_label,
        'survivor_label': surv_label, 'fail_cycle': fail_cycle,
        'fail_noise': fail_noise, 'calibration': cal_stats,
    })

    circuits_saved = {}

    print(f"{'Cyc':>3} {'Sel':>3} {'fid_A':>7} {'fid_B':>7} "
          f"{'DLP':>7} {'Static':>7} {'Greedy':>7} "
          f"{'p(A)':>6} {'p(B)':>6} {'Loss':>8} {'λ_A':>6} {'λ_B':>6}  Phase")
    print("-" * 95)
    t0 = time.time()
    total_dlp_shots = 0

    for t in range(num_cycles):
        phase = "NORMAL" if t < fail_cycle else "FAILURE"
        nA, nB = noise_A[t], noise_B[t]

        # ---- Measure BOTH paths ----
        qc_a = build_ghz(theta, path='A', noise_angle=nA)
        qc_b = build_ghz(theta, path='B', noise_angle=nB)
        counts_a, tqc_a = runner.run(qc_a, shots)
        counts_b, tqc_b = runner.run(qc_b, shots)
        fid_A = ghz_fid(counts_a, shots)
        fid_B = ghz_fid(counts_b, shots)
        total_dlp_shots += 2 * shots

        # Save circuits + diagrams at key moments
        save_cycles = [0, fail_cycle - 1, fail_cycle, num_cycles - 1]
        if t in save_cycles:
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

        # ---- Static baseline: always HW-preferred path ----
        qc_static = build_ghz(theta, path=pref_label,
                               noise_angle=noise_A[t] if preferred == 0 else noise_B[t])
        counts_static, _ = runner.run(qc_static, shots)
        fid_static = ghz_fid(counts_static, shots)

        # ---- Round-robin ----
        rr_path = t % 2
        rr_label = 'A' if rr_path == 0 else 'B'
        qc_rr = build_ghz(theta, path=rr_label,
                           noise_angle=noise_A[t] if rr_path == 0 else noise_B[t])
        counts_rr, _ = runner.run(qc_rr, shots)
        fid_rr = ghz_fid(counts_rr, shots)

        fid_greedy = max(fid_A, fid_B)

        # ---- Log ----
        log['static_fid'].append(fid_static)
        log['rr_fid'].append(fid_rr)
        log['greedy_fid'].append(fid_greedy)
        log['dlp_fid'].append(dlp_fid)
        log['dlp_path'].append(sel)
        log['pA'].append(pA); log['pB'].append(pB)
        log['fid_A'].append(fid_A); log['fid_B'].append(fid_B)
        log['loss'].append(l_val)
        log['logit_A'].append(lA); log['logit_B'].append(lB)
        log['counts_A'].append(counts_a)
        log['counts_B'].append(counts_b)
        log['counts_static'].append(counts_static)
        log['counts_rr'].append(counts_rr)

        sel_label = 'A' if sel == 0 else 'B'
        print(f"{t:3d}   {sel_label}   {fid_A:7.3f} {fid_B:7.3f}  "
              f"{dlp_fid:7.3f} {fid_static:7.3f} {fid_greedy:7.3f} "
              f"{pA:6.3f} {pB:6.3f} {l_val:8.4f} {lA:6.3f} {lB:6.3f}  {phase}")

    log['elapsed_sec'] = time.time() - t0
    log['total_dlp_shots'] = total_dlp_shots
    log['circuits_saved'] = circuits_saved
    return log


# ============================================================
#  Publication-quality figures
# ============================================================

def _fail_decor(ax, fc, N):
    """Add failure region shading and vertical line."""
    ax.axvline(fc, color='#D6604D', ls=':', alpha=0.6, lw=0.8)
    ax.axvspan(fc - 0.5, N - 0.5, alpha=0.04, color='#D6604D')


def plot_fidelity(log, filename):
    N = log['num_cycles']
    fc = log['fail_cycle']
    pref = log['preferred_label']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 2.2))

    _fail_decor(ax, fc, N)
    ax.plot(cycles, log['fid_A'], ':', color='#4393C3', lw=0.8, alpha=0.5, label='Path A (meas.)')
    ax.plot(cycles, log['fid_B'], ':', color='#F4A582', lw=0.8, alpha=0.5, label='Path B (meas.)')
    ax.plot(cycles, log['static_fid'], '--', color='#D6604D', lw=1.0, alpha=0.8,
            label=f'Static (always {pref})')
    ax.plot(cycles, log['greedy_fid'], '-', color='0.55', lw=0.8, alpha=0.6, label='Greedy oracle')
    ax.plot(cycles, log['dlp_fid'], '-s', color='#2CA02C', lw=1.5, ms=4, zorder=5, label='DLP Router')

    ax.set_ylabel('GHZ Fidelity')
    ax.set_xlabel('Calibration Cycle')
    ax.set_ylim([0, 1.08]); ax.set_xlim([-0.3, N - 0.7]); ax.set_xticks(cycles)
    ax.legend(loc='lower left', fontsize=6.5, ncol=2, framealpha=0.9,
              handlelength=1.8, columnspacing=1.0)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_probabilities(log, filename):
    N = log['num_cycles']
    fc = log['fail_cycle']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.5))

    _fail_decor(ax, fc, N)
    ax.plot(cycles, log['pA'], '-o', color='#4393C3', lw=1.3, ms=3.5, label='$p_A$')
    ax.plot(cycles, log['pB'], '-o', color='#F4A582', lw=1.3, ms=3.5, label='$p_B$')
    ax.axhline(0.5, color='grey', ls=':', alpha=0.3, lw=0.5)

    ax.set_ylabel('Path Probability $p_i$')
    ax.set_xlabel('Calibration Cycle')
    ax.set_ylim([-0.05, 1.08]); ax.set_xlim([-0.3, N - 0.7]); ax.set_xticks(cycles)
    ax.legend(loc='center right', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_noise_profile(log, filename):
    N = log['num_cycles']
    fc = log['fail_cycle']
    pref = log['preferred_label']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.3))

    _fail_decor(ax, fc, N)
    ax.plot(cycles, log['noise_A'], '-', color='#4393C3', lw=1.2, label='Path A noise')
    ax.plot(cycles, log['noise_B'], '-', color='#F4A582', lw=1.2, label='Path B noise')

    ax.set_ylabel('$R_x$ noise (rad)')
    ax.set_xlabel('Cycle')
    ax.set_xlim([-0.3, N - 0.7]); ax.set_xticks(cycles)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_logits(log, filename):
    N = log['num_cycles']
    fc = log['fail_cycle']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.5))

    _fail_decor(ax, fc, N)
    ax.plot(cycles, log['logit_A'], '-o', color='#4393C3', lw=1.3, ms=3.5, label='$\\lambda_A$')
    ax.plot(cycles, log['logit_B'], '-o', color='#F4A582', lw=1.3, ms=3.5, label='$\\lambda_B$')
    ax.axhline(0, color='grey', ls=':', alpha=0.3, lw=0.5)

    ax.set_ylabel('Structural Logit $\\lambda_i$')
    ax.set_xlabel('Calibration Cycle')
    ax.set_xlim([-0.3, N - 0.7]); ax.set_xticks(cycles)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_loss(log, filename):
    N = log['num_cycles']
    fc = log['fail_cycle']
    cycles = np.arange(N)
    fig, ax = plt.subplots(figsize=(COLW, 1.5))

    _fail_decor(ax, fc, N)
    ax.plot(cycles, log['loss'], '-s', color='#2CA02C', lw=1.3, ms=3.5)
    ax.set_ylabel('Loss $\\mathcal{L}$')
    ax.set_xlabel('Calibration Cycle')
    ax.set_xlim([-0.3, N - 0.7]); ax.set_xticks(cycles)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def plot_combined(log, filename):
    N = log['num_cycles']
    fc = log['fail_cycle']
    pref = log['preferred_label']
    cycles = np.arange(N)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DBLW, 3.2), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})

    for ax in [ax1, ax2]:
        _fail_decor(ax, fc, N)

    ax1.plot(cycles, log['fid_A'], ':', color='#4393C3', lw=0.8, alpha=0.5, label='Path A (meas.)')
    ax1.plot(cycles, log['fid_B'], ':', color='#F4A582', lw=0.8, alpha=0.5, label='Path B (meas.)')
    ax1.plot(cycles, log['static_fid'], '--', color='#D6604D', lw=1.0, alpha=0.8,
             label=f'Static (always {pref})')
    ax1.plot(cycles, log['greedy_fid'], '-', color='0.55', lw=0.8, alpha=0.6, label='Greedy oracle')
    ax1.plot(cycles, log['dlp_fid'], '-s', color='#2CA02C', lw=1.5, ms=4, zorder=5, label='DLP Router')
    ax1.set_ylabel('GHZ Fidelity')
    ax1.set_ylim([0, 1.08]); ax1.set_xlim([-0.3, N - 0.7])
    ax1.legend(loc='lower left', fontsize=6.5, ncol=3, framealpha=0.9,
               handlelength=1.8, columnspacing=1.0)
    ax1.grid(True, alpha=0.25)

    ax2.plot(cycles, log['pA'], '-o', color='#4393C3', lw=1.3, ms=3.5, label='$p_A$')
    ax2.plot(cycles, log['pB'], '-o', color='#F4A582', lw=1.3, ms=3.5, label='$p_B$')
    ax2.axhline(0.5, color='grey', ls=':', alpha=0.3, lw=0.5)
    ax2.set_ylabel('$p_i$')
    ax2.set_xlabel('Calibration Cycle')
    ax2.set_ylim([-0.05, 1.08]); ax2.set_xticks(cycles)
    ax2.legend(loc='center right', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT}{filename}.{ext}')
    plt.close()


def print_summary(log):
    N = log['num_cycles']
    fc = log['fail_cycle']
    pref = log['preferred_label']
    surv = log['survivor_label']
    pre, post = slice(0, fc), slice(fc, N)

    print(f"\n{'='*65}")
    print(f"RESULTS: Catastrophic Failure ({log['backend']})")
    print(f"{'='*65}")
    print(f"  HW-preferred:  Path {pref} (cal: A={log['calibration']['avg_A']:.4f}, "
          f"B={log['calibration']['avg_B']:.4f})")
    print(f"  Failure:       cycle {fc}+ on Path {pref}")
    print(f"")
    print(f"  --- PRE-FAILURE (cycles 0–{fc-1}) ---")
    print(f"  DLP:    {np.mean(log['dlp_fid'][pre]):.4f}")
    print(f"  Static: {np.mean(log['static_fid'][pre]):.4f}")
    print(f"  Paths:  {['A' if p==0 else 'B' for p in log['dlp_path'][pre]]}")
    print(f"")
    print(f"  --- POST-FAILURE (cycles {fc}–{N-1}) ---")
    print(f"  DLP:    {np.mean(log['dlp_fid'][post]):.4f}")
    print(f"  Static: {np.mean(log['static_fid'][post]):.4f}")
    print(f"  Paths:  {['A' if p==0 else 'B' for p in log['dlp_path'][post]]}")
    print(f"  Δ:      {(np.mean(log['dlp_fid'][post])-np.mean(log['static_fid'][post]))*100:+.1f} pp")
    print(f"")
    print(f"  --- OVERALL ---")
    print(f"  DLP:    {np.mean(log['dlp_fid']):.4f}")
    print(f"  Static: {np.mean(log['static_fid']):.4f}")
    print(f"  Greedy: {np.mean(log['greedy_fid']):.4f}")
    print(f"  DLP vs static: {(np.mean(log['dlp_fid'])-np.mean(log['static_fid']))*100:+.1f} pp")
    print(f"  Final logits:  λ_A={log['logit_A'][-1]:.4f}, λ_B={log['logit_B'][-1]:.4f}")
    print(f"  Circuits:      {len(log.get('circuits_saved',{}))} files in {OUT}circuits/")
    print(f"  Time:          {log['elapsed_sec']:.0f}s")
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
    print(f"CATASTROPHIC FAILURE (unbiased) on {runner.name}")
    print(f"  Phase 1: Router learns preferred path organically")
    print(f"  Phase 2: Catastrophic Rx noise on HW-preferred path")
    print(f"  DLP Softmax Router: L = 1 - Σ p_i·F_i")
    print(f"{'='*65}")

    log = run_catastrophic(
        runner,
        num_cycles=10,
        fail_cycle=5,
        fail_noise=1.2,
        shots=shots,
        train_iters=10,
        lr=0.5,
        n_cal_samples=5,
    )

    # ---- All figures ----
    plot_fidelity(log, f'fidelity_{tag}')
    plot_probabilities(log, f'probabilities_{tag}')
    plot_noise_profile(log, f'noise_{tag}')
    plot_logits(log, f'logits_{tag}')
    plot_loss(log, f'loss_{tag}')
    plot_combined(log, f'combined_{tag}')

    print_summary(log)

    # ---- Save JSON ----
    save_log = {k: v for k, v in log.items()}
    for k in ['counts_A', 'counts_B', 'counts_static', 'counts_rr']:
        save_log[k] = [dict(c) for c in save_log[k]]

    with open(f'{OUT}results_{tag}.json', 'w') as f:
        json.dump(save_log, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print(f"\nDONE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"All outputs in: {OUT}")