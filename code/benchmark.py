# benchmark.py
"""Benchmark PyTorch DataLoader vs NVIDIA DALI for images and video.

Examples:
  python benchmark.py image --data imagenette2-320/train --workers 4 8 16
  python benchmark.py image --data imagenette2-320/train --workers 8 --e2e
  python benchmark.py video --data videos_cfr/train --workers 4 8 --batch-size 8
  taskset -c 0-7 python benchmark.py image --data ... --workers 8   # simulate 8 cores/GPU

Every run appends rows to results.csv and prints a markdown table.
"""
import argparse, csv, os, random, statistics, threading, time
from datetime import datetime

import numpy as np
import psutil
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from train_dali import create_pipeline           # image pipeline (Step 4)
from step8_video import video_train_pipeline     # video pipeline (Step 7)


# --------------------------------------------------------------------------- #
# Utilization sampler: runs in a background thread while we measure
# --------------------------------------------------------------------------- #
class UtilSampler:
    def __init__(self, device=0, interval=0.1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        self.interval = interval
        self.gpu, self.cpu, self.mem = [], [], []
        self._stop = threading.Event()

    def _loop(self):
        psutil.cpu_percent(None)                       # prime the counter
        while not self._stop.is_set():
            self.gpu.append(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
            self.mem.append(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20)
            self.cpu.append(psutil.cpu_percent(None))  # % of ALL cores available
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        mean = lambda xs: statistics.mean(xs) if xs else float("nan")
        return mean(self.gpu), mean(self.cpu), max(self.mem, default=float("nan"))


def gpu_name():
    return torch.cuda.get_device_name(0)


# --------------------------------------------------------------------------- #
# Generic measurement loop
# --------------------------------------------------------------------------- #
def forever(loader):
    """Loop over epochs so that small datasets don't run out mid-benchmark."""
    while True:
        for batch in loader:
            yield batch


def measure(loader, get_xy, step, batch_size, warmup, iters, repeats):
    it = forever(loader)
    for _ in range(warmup):
        step(*get_xy(next(it)))
    torch.cuda.synchronize()

    runs = []
    for _ in range(repeats):
        with UtilSampler() as util:
            t0 = time.perf_counter()
            for _ in range(iters):
                step(*get_xy(next(it)))
            torch.cuda.synchronize()               # wait for queued GPU work
            elapsed = time.perf_counter() - t0
        g, c, m = util.summary()
        runs.append((iters * batch_size / elapsed, g, c, m))

    thr = [r[0] for r in runs]
    return {
        "samples_per_s": statistics.median(thr),
        "samples_per_s_std": statistics.stdev(thr) if len(thr) > 1 else 0.0,
        "gpu_util": statistics.median(r[1] for r in runs),
        "cpu_util": statistics.median(r[2] for r in runs),
        "peak_gpu_mem_mb": max(r[3] for r in runs),
    }


# --------------------------------------------------------------------------- #
# Steps: either "just touch the data" or a real training step
# --------------------------------------------------------------------------- #
def noop_step(x, y):
    pass


def make_train_step(model):
    model = model.cuda().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()

    def step(x, y):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return step


# --------------------------------------------------------------------------- #
# Image loaders
# --------------------------------------------------------------------------- #
def torch_image_loader(data, bs, workers):
    tf = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dl = DataLoader(ImageFolder(data, tf), batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=workers, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4)
    get_xy = lambda b: (b[0].cuda(non_blocking=True), b[1].cuda(non_blocking=True))
    return dl, get_xy


def dali_image_loader(data, bs, workers):
    pipe = create_pipeline(data_dir=data, is_training=True, crop=224, shard_id=0,
                           num_shards=1, batch_size=bs, num_threads=workers,
                           device_id=0, seed=0)
    pipe.build()
    it = DALIGenericIterator(pipe, ["data", "label"], reader_name="Reader",
                             last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)
    get_xy = lambda b: (b[0]["data"], b[0]["label"].squeeze(-1).long())
    return it, get_xy


# --------------------------------------------------------------------------- #
# Video loaders
# --------------------------------------------------------------------------- #
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class PyAVClipDataset(Dataset):
    """A typical CPU video loader: PyAV decode -> numpy -> torch resize/crop/flip.

    It seeks to the keyframe before a random start and decodes forward, which is the
    same strategy DALI uses, so the comparison is about decode hardware, not algorithm.
    """
    def __init__(self, root, seq_len=16, stride=2, crop=112):
        self.items = []
        for label, cls in enumerate(sorted(os.listdir(root))):
            for f in sorted(os.listdir(os.path.join(root, cls))):
                self.items.append((os.path.join(root, cls, f), label))
        self.seq_len, self.stride, self.crop = seq_len, stride, crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import av
        path, label = self.items[i]
        need = self.seq_len * self.stride
        with av.open(path) as container:
            s = container.streams.video[0]
            s.thread_type = "AUTO"
            fps = float(s.average_rate)
            n = s.frames or int(float(s.duration * s.time_base) * fps)
            start = random.randint(0, max(0, n - need))
            container.seek(int(start / fps / float(s.time_base)), stream=s)
            frames = []
            for frame in container.decode(s):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= need:
                    break
        while len(frames) < need:                   # pad very short videos
            frames.append(frames[-1])
        clip = torch.from_numpy(np.stack(frames[::self.stride][: self.seq_len]))  # F,H,W,C
        clip = clip.permute(0, 3, 1, 2).float() / 255.0                           # F,C,H,W
        clip = F.interpolate(clip, size=(128, 171), mode="bilinear", align_corners=False)
        y0 = random.randint(0, 128 - self.crop)
        x0 = random.randint(0, 171 - self.crop)
        clip = clip[..., y0:y0 + self.crop, x0:x0 + self.crop]
        if random.random() < 0.5:
            clip = clip.flip(-1)
        clip = clip.permute(1, 0, 2, 3)                                            # C,F,H,W
        return (clip - KINETICS_MEAN) / KINETICS_STD, label


def torch_video_loader(data, bs, workers, seq_len=16):
    dl = DataLoader(PyAVClipDataset(data, seq_len=seq_len), batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=workers, pin_memory=True, persistent_workers=True,
                    prefetch_factor=2)
    get_xy = lambda b: (b[0].cuda(non_blocking=True), b[1].cuda(non_blocking=True))
    return dl, get_xy


def dali_video_loader(data, bs, workers, seq_len=16):
    pipe = video_train_pipeline(video_root=data, seq_len=seq_len, batch_size=bs,
                                num_threads=workers, device_id=0, seed=0)
    pipe.build()
    it = DALIGenericIterator(pipe, ["video", "label"], reader_name="Reader",
                             last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)
    get_xy = lambda b: (b[0]["video"], b[0]["label"].squeeze(-1).long().cuda())
    return it, get_xy


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
LOADERS = {
    "image": {"pytorch": torch_image_loader, "dali": dali_image_loader},
    "video": {"pytorch": torch_video_loader, "dali": dali_video_loader},
}
MODELS = {
    "image": lambda: torchvision.models.resnet50(num_classes=1000),
    "video": lambda: torchvision.models.video.r2plus1d_18(num_classes=400),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modality", choices=["image", "video"])
    p.add_argument("--data", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[8])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=16, help="frames per clip (video only)")
    p.add_argument("--loaders", nargs="+", default=["pytorch", "dali"])
    p.add_argument("--e2e", action="store_true", help="include a real training step")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    bs = args.batch_size or (128 if args.modality == "image" else 8)
    torch.backends.cudnn.benchmark = True
    rows = []

    for workers in args.workers:
        for name in args.loaders:
            print(f"> {args.modality} | {name:7s} | workers={workers} | bs={bs} | "
                  f"{'e2e' if args.e2e else 'loader-only'}", flush=True)
            extra = {"seq_len": args.seq_len} if args.modality == "video" else {}
            loader, get_xy = LOADERS[args.modality][name](args.data, bs, workers, **extra)
            step = make_train_step(MODELS[args.modality]()) if args.e2e else noop_step
            res = measure(loader, get_xy, step, bs, args.warmup, args.iters, args.repeats)
            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "gpu": gpu_name(), "cpus": len(os.sched_getaffinity(0)),
                "modality": args.modality, "loader": name, "workers": workers,
                "batch_size": bs, "seq_len": args.seq_len if args.modality == "video" else "", "mode": "e2e" if args.e2e else "loader-only", **res,
            }
            rows.append(row)
            print(f"  {res['samples_per_s']:.0f} samples/s  "
                  f"(GPU {res['gpu_util']:.0f}%, CPU {res['cpu_util']:.0f}%, "
                  f"mem {res['peak_gpu_mem_mb']:.0f} MB)", flush=True)
            del loader                               # free workers / DALI pools
            torch.cuda.empty_cache()

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new_file:
            w.writeheader()
        w.writerows(rows)

    unit = "clips/s" if args.modality == "video" else "img/s"
    print(f"\n| loader | workers | {unit} | GPU util | CPU util | peak GPU mem |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['loader']} | {r['workers']} | {r['samples_per_s']:.0f} "
              f"± {r['samples_per_s_std']:.0f} | {r['gpu_util']:.0f}% | "
              f"{r['cpu_util']:.0f}% | {r['peak_gpu_mem_mb']:.0f} MB |")


if __name__ == "__main__":
    main()
