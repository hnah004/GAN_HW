"""End-to-end, frame-by-frame latency on the PYNQ-Z2 (run on the board, as root, next to generator.bit).
All processing is on-chip: PS does STFT/iSTFT in numpy, PL computes the mask.

    sudo python3 latency_bench_pynq.py --wav_dir noisy_wav --n_files 824 --out_dir enhanced

Per hop of 256 samples (16 ms) the following stages are timed:
    stft   : shift buffer, window, rfft, |.|, log1p                     (PS)
    pack   : float -> int16 (Q6.10), cache flush                         (PS)
    accel  : restart reg, DMA send+recv setup, ap_start, wait both DMA   (PS<->PL, includes PL compute)
    unpack : cache invalidate, int16 -> float                            (PS)
    istft  : expm1, gain, irfft, window, overlap-add                     (PS)
Enhanced wavs are written so PESQ of the complete on-board system can be computed on the PC."""
import argparse, gc, os, time
import numpy as np
from pynq import Overlay, allocate
import stream_dsp as sd

STAGES = ['stft', 'pack', 'accel', 'unpack', 'istft']

ap = argparse.ArgumentParser()
ap.add_argument('--bit', default='generator.bit')
ap.add_argument('--wav_dir', default='noisy_wav')
ap.add_argument('--out_dir', default='enhanced')
ap.add_argument('--n_files', type=int, default=824)
ap.add_argument('--scale', type=int, default=1024)     # data_t = ap_fixed<16,6> -> 10 fractional bits
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

ol = Overlay(args.bit)
dma, ip = ol.axi_dma_0, ol.generator_top_0
in_buf = allocate(shape=(257,), dtype=np.int16)
out_buf = allocate(shape=(257,), dtype=np.int16)
SCALE = args.scale
pc = time.perf_counter

files = sorted(f for f in os.listdir(args.wav_dir) if f.endswith('.wav'))[:args.n_files]
rec = []            # one row per frame: stft, pack, accel, unpack, istft
audio_sec = 0.0

for fname in files:
    x = sd.read_wav(os.path.join(args.wav_dir, fname))
    audio_sec += len(x) / sd.SR
    xp, T = sd.padded(x)
    st = sd.StreamSTFT()
    y = np.zeros(T * sd.HOP, np.float32)
    t_file = np.zeros((T, 5))
    gc.disable()
    for t in range(T):
        t0 = pc()
        spec, mag, logmag = st.analysis(xp[t * sd.HOP:(t + 1) * sd.HOP])
        t1 = pc()
        in_buf[:] = (logmag * SCALE).astype(np.int16)        # same conversion as run.ipynb
        in_buf.flush()
        t2 = pc()
        ip.write(0x10, 1 if t == 0 else 0)
        dma.sendchannel.transfer(in_buf)
        dma.recvchannel.transfer(out_buf)
        ip.write(0x00, 1)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        t3 = pc()
        out_buf.invalidate()
        mask = out_buf.astype(np.float32) / SCALE
        t4 = pc()
        hop_out = st.synthesis(spec, mag, logmag, mask)
        if t >= 1:
            y[(t - 1) * sd.HOP:t * sd.HOP] = hop_out
        t5 = pc()
        t_file[t] = (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
    gc.enable()
    y[(T - 1) * sd.HOP:len(x)] = st.flush(len(x) - (T - 1) * sd.HOP)
    sd.write_wav(os.path.join(args.out_dir, fname), y[:len(x)])
    rec.append(t_file)

in_buf.close(); out_buf.close()
R = np.concatenate(rec) * 1e3                      # ms
tot = R.sum(1)
np.savez('latency_frames.npz', R=R, stages=STAGES, files=files,
         frames_per_file=[len(r) for r in rec])

hop_ms, win_ms = 1e3 * sd.HOP / sd.SR, 1e3 * sd.N_FFT / sd.SR
print(f'files={len(files)} frames={len(tot)} audio={audio_sec:.1f}s')
print(f'{"stage":8s} {"mean":>7s} {"p50":>7s} {"p99":>7s} {"max":>7s}  (ms/frame)')
for i, s in enumerate(STAGES + ['TOTAL']):
    v = tot if s == 'TOTAL' else R[:, i]
    print(f'{s:8s} {v.mean():7.3f} {np.median(v):7.3f} {np.percentile(v, 99):7.3f} {v.max():7.3f}')
print(f'compute-only RTF = {tot.sum() / 1e3 / audio_sec:.3f}   throughput = {len(tot) / (tot.sum() / 1e3):.1f} frames/s')
print(f'frames over hop ({hop_ms:.0f} ms): {(tot > hop_ms).sum()}   over {40 - win_ms:.0f} ms: {(tot > 40 - win_ms).sum()}')
print(f'end-to-end latency = {win_ms:.0f} ms (algorithmic) + {tot.mean():.2f} ms mean / '
      f'{np.percentile(tot, 99):.2f} ms p99 processing = {win_ms + tot.mean():.2f} / {win_ms + np.percentile(tot, 99):.2f} ms')
