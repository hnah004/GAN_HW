"""Frame-by-frame STFT / iSTFT in numpy, equivalent to signal_processing.py
(torch.stft/istft, n_fft=512, hop=256, Hamming, center=True, log1p magnitude).
Runs on the PYNQ-Z2 ARM (numpy only)."""
import wave
import numpy as np

SR, N_FFT, HOP = 16000, 512, 256
# torch.hamming_window(512) is periodic by default
WIN = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
ENV = WIN[:HOP] ** 2 + WIN[HOP:] ** 2      # steady-state OLA normalisation (as torch.istft)
ENV_TAIL = WIN[HOP:] ** 2                  # last half-frame: only one frame contributes


class StreamSTFT:
    def __init__(self):
        self.reset()

    def reset(self):
        self.inbuf = np.zeros(N_FFT, np.float32)   # zeros = center=True constant padding
        self.ola = np.zeros(HOP, np.float32)

    def analysis(self, hop):
        """hop: 256 new samples -> (complex spec, |spec|, log1p|spec|), 257 bins."""
        self.inbuf[:HOP] = self.inbuf[HOP:]
        self.inbuf[HOP:] = hop
        spec = np.fft.rfft(self.inbuf * WIN)
        mag = np.abs(spec)
        return spec, mag, np.log1p(mag).astype(np.float32)

    def synthesis(self, spec, mag, logmag, mask):
        """Apply mask in log1p domain, keep noisy phase, overlap-add.
        Returns 256 finished output samples (delayed by one hop)."""
        enh = np.expm1(mask * logmag)
        gain = np.divide(enh, mag, out=np.zeros_like(enh), where=mag > 0)
        frame = np.fft.irfft(spec * gain, N_FFT).astype(np.float32) * WIN
        out = (self.ola + frame[:HOP]) / ENV
        self.ola = frame[HOP:].copy()
        return out

    def flush(self, n):
        return self.ola[:n] / ENV_TAIL[:n]


def n_frames(length):
    return 1 + length // HOP                    # same count as torch.stft(center=True)


def padded(x):
    T = n_frames(len(x))
    xp = np.zeros(T * HOP, np.float32)
    xp[:len(x)] = x
    return xp, T


def read_wav(path):
    with wave.open(path, 'rb') as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def write_wav(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1 - 1 / 32768) * 32768).astype(np.int16).tobytes())


def enhance_stream(x, mask_fn, timer=None):
    """Run the whole utterance hop by hop. mask_fn(logmag, first) -> mask[257].
    Output sample t is emitted after frame t//256 + 1, i.e. algorithmic delay = N_FFT."""
    st = StreamSTFT()
    xp, T = padded(x)
    out = np.zeros(T * HOP, np.float32)
    for t in range(T):
        spec, mag, logmag = st.analysis(xp[t * HOP:(t + 1) * HOP])
        mask = mask_fn(logmag, t == 0)
        y = st.synthesis(spec, mag, logmag, mask)
        if t >= 1:                                # frame 0's first half is the centre padding
            out[(t - 1) * HOP:t * HOP] = y
    tail = len(x) - (T - 1) * HOP
    out[(T - 1) * HOP:len(x)] = st.flush(tail)
    return out[:len(x)]
