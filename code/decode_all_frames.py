#!/usr/bin/env python
"""Decode EVERY frame of every video three ways and compare frames/s.

  cpu     : FFmpeg software decoding, N parallel processes (1 thread each)
  ffmpeg-gpu : FFmpeg with NVDEC (-hwaccel cuda), N parallel processes, frames stay on GPU
  dali    : DALI fn.readers.video (NVDEC), every frame (stride=1, non-overlapping
            sequences of `--seq-len` frames), frames stay on GPU

No resizing, cropping or augmentation: this isolates pure decode throughput.

  python decode_all_frames.py --data videos_1080/train --workers 8
"""
import argparse, os, re, statistics, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import pynvml


def list_videos(root):
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.join(dp, f) for f in fs if f.endswith((".mp4", ".mkv", ".mov"))]
    return sorted(out)


class DecoderMeter:
    """Samples NVDEC utilisation (%) in the background."""
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.vals, self.stop = [], threading.Event()

    def __enter__(self):
        def run():
            while not self.stop.is_set():
                self.vals.append(pynvml.nvmlDeviceGetDecoderUtilization(self.h)[0])
                time.sleep(0.2)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.stop.set(); self.t.join()

    @property
    def mean(self):
        return statistics.mean(self.vals) if self.vals else 0.0


def _ffmpeg(path, gpu):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "1"]
    if gpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += ["-i", path, "-an", "-progress", "pipe:1", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return int(re.findall(r"frame=(\d+)", r.stdout)[-1])


def run_ffmpeg(files, workers, gpu):
    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(lambda f: _ffmpeg(f, gpu), files))


def make_dali(data, workers, seq_len, batch_size):
    from nvidia.dali import pipeline_def, fn

    @pipeline_def
    def pipe_fn():
        frames, _ = fn.readers.video(
            device="gpu", file_root=data, sequence_length=seq_len, stride=1,
            step=seq_len,                 # consecutive, non-overlapping -> every frame once
            random_shuffle=False, name="Reader", skip_vfr_check=True)
        return frames

    pipe = pipe_fn(batch_size=batch_size, num_threads=workers, device_id=0, seed=0)
    pipe.build()
    n_seq = pipe.epoch_size("Reader")
    return pipe, n_seq


def run_dali(pipe, n_seq, seq_len, batch_size):
    import torch
    n_iter = n_seq // batch_size          # drop the last partial batch (< batch_size sequences)
    for _ in range(n_iter):
        pipe.run()
    torch.cuda.synchronize()
    return n_iter * batch_size * seq_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=256, help="DALI sequence length")
    ap.add_argument("--batch-size", type=int, default=2, help="DALI batch size (sequences per iteration)")
    ap.add_argument("--label", default="", help="resolution label written to the CSV, e.g. 1080p")
    ap.add_argument("--out", default="", help="append results to this CSV")
    ap.add_argument("--methods", nargs="+", default=["cpu", "ffmpeg-gpu", "dali"])
    a = ap.parse_args()

    files = list_videos(a.data)
    print(f"{len(files)} videos in {a.data}", flush=True)
    rows = []
    for m in a.methods:
        if m == "dali":
            pipe, n_seq = make_dali(a.data, a.workers, a.seq_len, a.batch_size)
            fn_run = lambda: run_dali(pipe, n_seq, a.seq_len, a.batch_size)
        else:
            fn_run = lambda m=m: run_ffmpeg(files, a.workers, gpu=(m == "ffmpeg-gpu"))
        fn_run()                                   # warm-up (page cache, CUDA context)
        rates, nvdec, frames = [], [], 0
        for _ in range(a.repeats):
            with DecoderMeter() as dm:
                t = time.time(); frames = fn_run(); dt = time.time() - t
            rates.append(frames / dt); nvdec.append(dm.mean)
        r = statistics.median(rates)
        rows.append((m, frames, r, statistics.pstdev(rates), statistics.mean(nvdec)))
        print(f"{m:11s} {frames:7d} frames  {r:8.0f} frames/s  (NVDEC {rows[-1][4]:.0f}%)", flush=True)

    if a.out:
        import csv
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["resolution", "method", "seq_len", "frames", "frames_per_s", "std", "nvdec_util"])
            for m, fr, r, sd, nv in rows:
                w.writerow([a.label, m, a.seq_len if m == "dali" else "", fr, round(r, 1), round(sd, 1), round(nv, 1)])

    base = rows[0][2]
    print(f"\n| method | frames decoded | frames/s | NVDEC util | vs CPU |\n|---|---|---|---|---|")
    for m, fr, r, sd, nv in rows:
        print(f"| {m} | {fr} | {r:,.0f} ± {sd:,.0f} | {nv:.0f}% | {r / base:.1f}× |")


if __name__ == "__main__":
    main()
