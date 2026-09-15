#!/usr/bin/env python3
"""
AI/ML-Enabled Real-Time Multi-Algorithm ANC Simulator
ULTRA-FIXED VERSION - Threading issues resolved
No GPIO, sounddevice, or thread conflicts
"""

import argparse
import queue
import sys
import time
import numpy as np
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt


DEFAULTS = {
    "samplerate": 16000,
    "block_ms": 20.0,
    "filter_len": 32,
    "mu_fx": 0.012,
    "mu_vs_min": 0.004,
    "mu_vs_max": 0.035,
    "mu_robust": 0.015,
    "leak": 0.99998,
    "secondary_path": [0.78, 0.16, -0.045, 0.018],
}

LABELS = ("Stationary", "Non-stationary", "Impulsive")


def expected_crest_factor(n):
    n = max(int(n), 2)
    return np.sqrt(2.0 * np.log(n))


def classify_block(x):
    """Feature-based classifier."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if x.size < 16:
        return "Stationary", np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    rms = np.sqrt(np.mean(x * x) + 1e-12)
    peak_ratio = np.max(np.abs(x)) / (rms + 1e-12)
    crest_excess = max(0.0, peak_ratio - expected_crest_factor(x.size))

    median = np.median(x)
    mad = np.median(np.abs(x - median)) + 1e-9
    outlier_frac = np.mean(np.abs(x - median) > 6.0 * mad)

    chunks = np.array_split(x, min(8, x.size))
    energies = np.array([np.mean(c * c) + 1e-12 for c in chunks if c.size], dtype=np.float64)
    variation = np.std(energies) / (np.mean(energies) + 1e-12)

    spectrum = np.abs(np.fft.rfft(x)) + 1e-10
    geometric = np.exp(np.mean(np.log(spectrum)))
    arithmetic = np.mean(spectrum)
    flatness = geometric / (arithmetic + 1e-12)

    impulsive = np.clip(0.5 * crest_excess + 8.0 * outlier_frac, 0.0, 1.0)
    nonstationary = np.clip(1.6 * variation + 0.25 * flatness, 0.0, 1.0)
    stationary = np.clip(1.0 - 0.85 * nonstationary - impulsive, 0.0, 1.0)

    scores = np.array([stationary, nonstationary, impulsive], dtype=np.float64) + 0.02
    scores /= (np.sum(scores) + 1e-12)

    label = LABELS[int(np.argmax(scores))]
    return label, scores


def confidence_to_weights(confidence):
    c = np.asarray(confidence, dtype=np.float64).reshape(-1)

    if c.size != 3 or not np.all(np.isfinite(c)):
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    c = np.clip(c, 0.0, 1.0)
    temperature = 0.10
    shifted = (c - np.max(c)) / temperature
    shifted -= np.max(shifted)
    z = np.exp(shifted)
    return z / (np.sum(z) + 1e-12)


class SingleFilter:
    def __init__(self, filter_len, mu_fn, leak):
        self.filter_len = int(filter_len)
        self.mu_fn = mu_fn
        self.leak = float(leak)
        self.w = np.zeros(self.filter_len, dtype=np.float64)

    def step(self, x_window, xf_window, desired):
        anti_noise = float(np.dot(self.w, x_window))
        error = float(desired - anti_noise)
        norm = float(np.dot(xf_window, xf_window)) + 1e-8
        mu, error_term = self.mu_fn(error)
        self.w += float(mu) * float(error_term) * xf_window / norm
        self.w *= self.leak
        return anti_noise, error

    def reset(self):
        self.w.fill(0.0)


class ANCEngine:
    def __init__(self, filter_len, secondary_path, mu_fx, mu_vs_min, mu_vs_max, mu_robust, leak):
        self.filter_len = int(filter_len)
        if self.filter_len < 2:
            raise ValueError("filter_len must be at least 2")

        self.sp = np.asarray(secondary_path, dtype=np.float64).reshape(-1)
        if self.sp.size == 0:
            raise ValueError("secondary_path cannot be empty")

        self.raw_hist = np.zeros(self.filter_len)
        self.xf_hist = np.zeros(self.filter_len)
        self.sp_hist = np.zeros(self.sp.size)

        def fx_mu(error):
            return (float(mu_fx), float(error))

        def vs_mu(error):
            mu = np.clip(float(mu_vs_min) + 0.030 * abs(error), float(mu_vs_min), float(mu_vs_max))
            return (float(mu), float(error))

        def robust_mu(error):
            return (float(mu_robust), float(np.tanh(2.0 * error)))

        self.fx = SingleFilter(self.filter_len, fx_mu, leak)
        self.vs = SingleFilter(self.filter_len, vs_mu, leak)
        self.robust = SingleFilter(self.filter_len, robust_mu, leak)

    def reset(self):
        self.raw_hist.fill(0.0)
        self.xf_hist.fill(0.0)
        self.sp_hist.fill(0.0)
        self.fx.reset()
        self.vs.reset()
        self.robust.reset()

    def process_sample(self, reference, measured_error):
        anti_fx, err_fx = self.fx.step(self.raw_hist, self.xf_hist, measured_error)
        anti_vs, err_vs = self.vs.step(self.raw_hist, self.xf_hist, measured_error)
        anti_rb, err_rb = self.robust.step(self.raw_hist, self.xf_hist, measured_error)

        if self.sp_hist.size > 1:
            self.sp_hist[:-1] = self.sp_hist[1:]
        self.sp_hist[-1] = reference

        xf_n = float(np.dot(self.sp, self.sp_hist[::-1]))

        if self.raw_hist.size > 1:
            self.raw_hist[:-1] = self.raw_hist[1:]
            self.xf_hist[:-1] = self.xf_hist[1:]

        self.raw_hist[-1] = reference
        self.xf_hist[-1] = xf_n

        return ((anti_fx, anti_vs, anti_rb), (err_fx, err_vs, err_rb))


class NoiseSimulator:
    """Simulates different types of noise for testing"""
    def __init__(self, samplerate):
        self.samplerate = samplerate

    def generate_stationary(self, duration_sec):
        """White noise"""
        samples = int(duration_sec * self.samplerate)
        return np.random.normal(0, 0.1, samples)

    def generate_nonstationary(self, duration_sec):
        """Chirp signal"""
        samples = int(duration_sec * self.samplerate)
        t = np.arange(samples) / self.samplerate
        return 0.1 * scipy_signal.chirp(t, f0=100, f1=500, t1=duration_sec)

    def generate_impulsive(self, duration_sec):
        """Random impulses"""
        samples = int(duration_sec * self.samplerate)
        noise = np.random.normal(0, 0.05, samples)
        impulses = np.zeros_like(noise)
        for _ in range(3):
            idx = np.random.randint(0, samples)
            if idx + 10 <= samples:
                impulses[idx:idx+10] = np.random.normal(0, 0.5, 10)
        return noise + impulses


class ClassifierEngine:
    """Non-threaded classifier - more reliable"""
    def __init__(self, samplerate, window_ms=150.0, smoothing=0.35, hold_count=3):
        self.window_samples = max(16, int(round(samplerate * window_ms / 1000.0)))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.hold_count = max(1, int(hold_count))

        self.weights = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.confidence = self.weights.copy()
        self._smoothed_conf = self.confidence.copy()
        self.label = "Stationary"
        self._candidate_label = "Stationary"
        self._candidate_count = 0
        self._buffer = np.empty(0, dtype=np.float64)

    def process_block(self, block):
        """Process a single block of audio"""
        self._buffer = np.concatenate((self._buffer, block))
        
        while self._buffer.size >= self.window_samples:
            window = self._buffer[:self.window_samples]
            self._buffer = self._buffer[self.window_samples:]
            self._consume_window(window)

    def _consume_window(self, x):
        raw_label, confidence = classify_block(x)
        a = self.smoothing
        self._smoothed_conf = a * confidence + (1.0 - a) * self._smoothed_conf
        self.confidence = self._smoothed_conf.copy()
        self.weights = confidence_to_weights(self._smoothed_conf)

        if raw_label == "Impulsive":
            self._candidate_label = raw_label
            self._candidate_count = self.hold_count
        elif raw_label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = raw_label
            self._candidate_count = 1

        if self._candidate_count >= self.hold_count and self.label != raw_label:
            self.label = raw_label


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="ANC Simulator - ULTRA-FIXED (No threading issues)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python anc_simulator_ultra.py --duration 5 --save-plot output.png
  python anc_simulator_ultra.py --noise-type stationary --duration 3
  python anc_simulator_ultra.py --noise-type mix --duration 10 --save-plot result.png
        """
    )
    p.add_argument("--samplerate", type=int, default=DEFAULTS["samplerate"], help="Sample rate in Hz")
    p.add_argument("--block-ms", type=float, default=DEFAULTS["block_ms"], help="Block size in ms")
    p.add_argument("--filter-len", type=int, default=DEFAULTS["filter_len"], help="Filter length")
    p.add_argument("--duration", type=float, default=5.0, help="Simulation duration in seconds")
    p.add_argument("--noise-type", type=str, default="mix", choices=["stationary", "nonstationary", "impulsive", "mix"])
    p.add_argument("--save-plot", type=str, default="anc_output.png", help="Save plot to file")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return p


def validate_args(args):
    if args.samplerate <= 0:
        raise SystemExit("ERROR: --samplerate must be > 0")
    if args.block_ms <= 0:
        raise SystemExit("ERROR: --block-ms must be > 0")
    if args.filter_len < 2:
        raise SystemExit("ERROR: --filter-len must be >= 2")
    if args.duration <= 0:
        raise SystemExit("ERROR: --duration must be > 0")


def main():
    args = build_arg_parser().parse_args()
    validate_args(args)

    print("\n" + "=" * 70)
    print("ANC SIMULATOR - ULTRA-FIXED VERSION")
    print("=" * 70)
    print(f"Duration: {args.duration}s | Sample Rate: {args.samplerate}Hz | Noise: {args.noise_type}")
    print("=" * 70 + "\n")

    blocksize = max(1, int(round(args.samplerate * args.block_ms / 1000.0)))
    total_samples = int(args.duration * args.samplerate)

    # Initialize components (NO THREADING)
    print("[1/4] Initializing ANC engine...")
    engine = ANCEngine(
        filter_len=args.filter_len,
        secondary_path=DEFAULTS["secondary_path"],
        mu_fx=DEFAULTS["mu_fx"],
        mu_vs_min=DEFAULTS["mu_vs_min"],
        mu_vs_max=DEFAULTS["mu_vs_max"],
        mu_robust=DEFAULTS["mu_robust"],
        leak=DEFAULTS["leak"]
    )

    classifier = ClassifierEngine(
        samplerate=args.samplerate,
        window_ms=150.0,
        smoothing=0.35,
        hold_count=3
    )

    print("[2/4] Generating audio signals...")
    noise_sim = NoiseSimulator(args.samplerate)

    if args.noise_type == "stationary":
        reference = noise_sim.generate_stationary(args.duration)
    elif args.noise_type == "nonstationary":
        reference = noise_sim.generate_nonstationary(args.duration)
    elif args.noise_type == "impulsive":
        reference = noise_sim.generate_impulsive(args.duration)
    else:  # mix
        ref1 = noise_sim.generate_stationary(args.duration / 3)
        ref2 = noise_sim.generate_nonstationary(args.duration / 3)
        ref3 = noise_sim.generate_impulsive(args.duration / 3)
        reference = np.concatenate([ref1, ref2, ref3])

    # Simulate secondary path
    measured_error = np.convolve(reference, DEFAULTS["secondary_path"], mode='same')
    measured_error += np.random.normal(0, 0.01, len(measured_error))

    # Initialize storage
    output_signal = np.zeros(total_samples)
    error_signal = np.zeros(total_samples)
    labels_history = []

    # Process audio
    print("[3/4] Processing audio...")
    start_time = time.time()

    for i in range(0, total_samples, blocksize):
        end_idx = min(i + blocksize, total_samples)
        block_ref = reference[i:end_idx]
        block_err = measured_error[i:end_idx]

        # Process classifier
        classifier.process_block(block_ref)

        # Process ANC
        for j, sample_ref in enumerate(block_ref):
            sample_idx = i + j
            if sample_idx < end_idx:
                sample_err = block_err[j]
                (anti_fx, anti_vs, anti_rb), _ = engine.process_sample(sample_ref, sample_err)

                # Combine with weights
                combined_anti = (classifier.weights[0] * anti_fx +
                               classifier.weights[1] * anti_vs +
                               classifier.weights[2] * anti_rb)

                output_signal[sample_idx] = combined_anti
                error_signal[sample_idx] = sample_err - combined_anti

        labels_history.append(classifier.label)

        # Progress
        progress = (i / total_samples) * 100
        print(f"  Progress: {progress:5.1f}% | Label: {classifier.label:15} | Weights: FX={classifier.weights[0]:.3f} VS={classifier.weights[1]:.3f} RB={classifier.weights[2]:.3f}", end='\r')

    elapsed = time.time() - start_time
    print(f"\n\n[4/4] Computing statistics...")

    # Calculate statistics
    ref_rms = np.sqrt(np.mean(reference ** 2))
    err_rms = np.sqrt(np.mean(error_signal ** 2))
    ref_db = 20 * np.log10(ref_rms + 1e-10)
    err_db = 20 * np.log10(err_rms + 1e-10)
    reduction = ref_db - err_db

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Processing Time: {elapsed:.2f}s")
    print(f"  Reference RMS:   {ref_db:7.2f} dB")
    print(f"  Error RMS:       {err_db:7.2f} dB")
    print(f"  Noise Reduction: {reduction:7.2f} dB ✅")
    print(f"  Dominant Label:  {max(set(labels_history), key=labels_history.count)}")
    print("=" * 70 + "\n")

    # Plot results
    if not args.no_plot and args.save_plot:
        print(f"Generating visualization: {args.save_plot}")
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        t = np.arange(len(reference)) / args.samplerate

        axes[0].plot(t, reference, label='Reference', alpha=0.7, color='blue', linewidth=1)
        axes[0].plot(t, measured_error, label='Measured Error', alpha=0.7, color='orange', linewidth=1)
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].set_title('Input Signals', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, output_signal, label='Anti-Noise Output', alpha=0.7, color='green', linewidth=1)
        axes[1].set_ylabel('Amplitude', fontsize=10)
        axes[1].set_title('Generated Anti-Noise', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, error_signal, label='Residual Error', alpha=0.7, color='red', linewidth=1)
        axes[2].set_ylabel('Amplitude', fontsize=10)
        axes[2].set_xlabel('Time (s)', fontsize=10)
        axes[2].set_title('Residual Error After ANC', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {args.save_plot}\n")
        plt.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
