# HW/SW Co-Design of a GAN-Based Speech Enhancement on FPGA

## Results
VoiceBank-DEMAND test set (824 utterances), full pipeline running on the board:

| PESQ | CSIG | CBAK | COVL | Params |
|---|---|---|---|---|
| 2.93 | 3.99 | 3.02 | 3.45 | 0.38 M |

Per-frame processing time (hop = 16 ms):

| Stage | Unit | Mean (ms) | P99 (ms) |
|---|---|---|---|
| STFT + log-magnitude | PS | 0.55 | 0.72 |
| Pack + cache flush | PS | 0.27 | 0.39 |
| DMA + accelerator | PS↔PL | 4.59 | 4.79 |
| Unpack | PS | 0.45 | 0.68 |
| Mask + iSTFT + overlap-add | PS | 1.79 | 2.42 |
| **Total** | | **7.64** | **8.89** |

- Real-time factor: 0.48 (130 frames/s end-to-end)
- End-to-end latency: 32 ms window + 7.64 ms processing = 39.6 ms


## Repository structure

```
training/            Training (PyTorch)
hls/                 Vitis HLS source of the generator IP + fixed-point weights and LUTs
vivado/              Block design and implementation reports (utilization, timing, power)
pynq/overlay/        Prebuilt bitstream (.bit) and hardware handoff (.hwh)
pynq/latency/        Streaming STFT/iSTFT and on-board latency benchmark
results/board_run/   Benchmark log and per-utterance scores from the board
```

## Usage

### 1. Dataset
Download [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) and place the 48 kHz files in
`data/VCTK_DEMAND_48k/{train,test}/{clean,noisy}/`. Then resample to 16 kHz:
```
cd training
python downsample.py
```

### 2. Training
```
cd training
pip install -r ../requirements.txt
python main.py --exp_name causal128 --base_path data/VCTK_DEMAND
python inference.py --weight_path results/causal128/model --weight_file best_model.pth \
                    --base_path data/VCTK_DEMAND
```

### 3. Hardware
- HLS: create a Vitis HLS project from `hls/lstm_forward.cpp`
  (top function `generator_top`, part `xc7z020clg400-1`) and export the IP.
- Vivado: connect the IP to the Zynq PS through AXI DMA as in `vivado/design_1.bd`.
- A prebuilt bitstream is provided in `pynq/overlay/`.


### 4. Running on the PYNQ-Z2
Copy `pynq/overlay/*`, `pynq/latency/*` and the noisy test wavs to the board, then run:
```
sudo python3 latency_bench_pynq.py --bit generator.bit --wav_dir noisy_wav --out_dir enhanced
```
The script processes every utterance frame by frame on the board, prints the latency breakdown,
and writes the enhanced wavs to `enhanced/`.





