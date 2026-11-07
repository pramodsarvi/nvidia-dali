# Feeding the Beast: An End-to-End Guide to NVIDIA DALI

*How to stop your GPU from waiting for your CPU, with a hands-on tutorial from first pipeline to multi-GPU training.*

---

## TL;DR

I ran every script in this tutorial on one NVIDIA L40S, limited to 8 CPU cores (one GPU's usual share of a server). Full details are in section 13.6.

| Workload | PyTorch | DALI | Speedup |
|---|---|---|---|
| Images, data loading only (8 workers) | 2,805 img/s | 19,086 img/s | **6.8×** |
| Images, data loading only (2 workers) | 952 img/s | 17,781 img/s | **18.7×** |
| Images, ResNet-50 training | 1,110 img/s | 1,204 img/s | 1.08× |
| Video clips, data loading only (8 workers) | 96 clips/s | 126 clips/s | **1.3×** |
| Video clips, data loading only (4 workers) | 65 clips/s | 126 clips/s | **1.9×** |
| Video clips, R(2+1)D-18 training | 62 clips/s | 109 clips/s | **1.8×** |

Two things to take from this table:

1. **DALI removes the data-loading bottleneck.** Loading is 7–19× faster for images and 1.3–1.9× faster for video, and it barely needs CPU cores (2–4 workers are already enough).
2. **That only speeds up training if loading was the bottleneck.** ResNet-50 on this GPU is already limited by the model, so image training is only 8% faster. The video model is much more demanding on the input side, so training gets 1.8× faster. Section 13.7 explains how to tell which case you are in.

---

## 1. The problem: your GPU is starving

Modern GPUs can process thousands of images per second. A typical PyTorch input pipeline looks like this:

```
disk → read JPEG → decode (CPU) → resize/crop (CPU) → flip (CPU) → normalize (CPU) → to tensor → copy to GPU → model
```

Everything before the copy runs on the CPU, inside Python `DataLoader` workers. JPEG decoding and resizing are expensive. On an 8-GPU server with 32–64 CPU cores, the CPU often becomes the bottleneck. You can see it when:

- `nvidia-smi` shows GPU utilization going up and down (for example 40% → 100% → 30%).
- Adding `num_workers` stops helping, or fills up RAM.
- A faster GPU makes training only slightly faster.

**NVIDIA DALI (Data Loading Library)** fixes this by moving decoding and augmentation onto the GPU. It also runs everything in an optimized C++ engine with prefetching, outside the Python GIL.

```
disk → read (CPU) → decode (nvJPEG / hardware decoder, GPU) → augment (GPU) → model
```

## 2. Core concepts

Learn these five ideas and the rest of DALI is straightforward:

| Concept | What it means |
|---|---|
| **Pipeline** | A graph of data-processing operations. You define it once. DALI builds it and runs it asynchronously. |
| **Operators (`fn.*`)** | The building blocks: readers, decoders, resize, crop, normalize, random number generators, and more. |
| **Device placement** | Each operator runs on `"cpu"` or `"gpu"`. `"mixed"` means CPU input and GPU output (used by decoders). Data can move CPU → GPU but **never back** inside a pipeline. |
| **Batches** | DALI always works on a batch of samples. A batch can contain samples of different sizes (for example images of different resolutions). |
| **Iterators / plugins** | Adapters that turn pipeline outputs into PyTorch, TensorFlow, JAX, or PaddlePaddle tensors without copying. |

Important: the code inside a `@pipeline_def` function does **not** process data when Python runs it. It builds a graph. Think of it as describing the pipeline, not executing it.

## 3. Under the hood: how DALI actually runs

Knowing the execution model helps when you tune performance or debug errors later.

### 3.1 Graph construction vs execution

When you call a `@pipeline_def` function, each `fn.*` call returns a **`DataNode`**. This is a symbolic handle, not data. Python operators work on these handles too: `images * 0.5 + 10`, `fn.cast(x, dtype=types.FLOAT)`, comparisons, and so on. All of them add nodes to the graph. When you call `pipe.build()`:

1. DALI checks the graph and infers the device of each node.
2. It removes operators whose outputs are never used.
3. It splits the graph into **three stages**: CPU → mixed → GPU.
4. It allocates the buffers and memory pools for each stage.

After that, `pipe.run()` (or the iterator) only **pops finished batches from a queue**. The Python code you wrote is not called again.

### 3.2 The three-stage asynchronous executor

```
            ┌──────────── CPU thread pool (num_threads) ────────────┐
 batch N+2: │ readers, CPU ops (file read, CPU decode, python fns)   │
            └───────────────────────────┬───────────────────────────┘
                                        ▼
 batch N+1: │ mixed ops (decoders: host bytes → nvJPEG/HW → device)  │
                                        ▼
 batch N+1: │ GPU ops on a dedicated CUDA stream (resize, CMN, …)    │
                                        ▼
 batch N  : │ output queue (prefetch_queue_depth) → your training    │
```

- **Stages overlap.** While the GPU stage works on batch *N+1*, the CPU stage is already reading batch *N+2*, and your model is training on batch *N*.
- **The CPU stage is parallel within a batch.** `num_threads` workers split the samples of the batch between them. This runs in C++, so the Python GIL is not involved.
- **GPU work runs on DALI's own CUDA stream.** Before giving you a tensor, the PyTorch plugin synchronizes that stream with PyTorch's current stream, so you never read a half-written buffer.
- **`prefetch_queue_depth`** (default 2) sets how many complete batches are buffered. A larger value smooths out uneven batch times (some JPEGs are much bigger than others) but uses more GPU memory. With `exec_pipelined=True, exec_async=True` (the defaults) you can set the CPU and GPU depths separately: `prefetch_queue_depth={"cpu_size": 4, "gpu_size": 2}` together with `exec_separated=True`.

### 3.3 Inside the decoder

`fn.decoders.image(device="mixed")` is a **hybrid** decoder:

1. **Parse** (CPU): reads the JPEG headers and Huffman tables.
2. **Huffman decode**: done on the CPU for small images and on the GPU for large ones. DALI decides automatically per image, and `hybrid_huffman_threshold` controls the cutoff.
3. **IDCT + colour conversion** (GPU): the most compute-heavy part.
4. **Hardware decoder**: A100, H100 and newer GPUs have a dedicated JPEG engine. `hw_decoder_load=0.65` sends about 65% of each batch to that engine and the rest to the CUDA path, so both run at the same time. Tune this value per GPU.

Formats other than JPEG (PNG, TIFF, BMP, WebP, JPEG 2000) fall back to CPU or nvJPEG2000 paths automatically. The pipeline code doesn't change.

### 3.4 Memory model

DALI uses its own **memory pools** for both GPU and pinned host memory. Because batches can contain samples of different sizes, buffers start small and **grow to the largest batch seen so far**. They do not shrink afterwards. This explains two things you'll notice:

- GPU memory use goes up during the first few hundred iterations and then stays flat.
- One unusually large image can raise memory use for the rest of the run.

To get stable memory use from the first iteration, give DALI size hints:

```python
images = fn.decoders.image(
    jpegs, device="mixed",
    preallocate_width_hint=5980, preallocate_height_hint=6430,  # largest image in the dataset
    device_memory_padding=211025920,      # extra GPU scratch memory for nvJPEG
    host_memory_padding=140544512,        # extra pinned host memory
)
```

(These numbers are the values NVIDIA uses in its ImageNet examples.)

### 3.5 Per-sample vs per-batch arguments

Every operator argument can be either:
- a **constant**, which is the same for every sample (`resize_x=224`), or
- a **`DataNode`** from another operator, which gives one value per sample (`angle=fn.random.uniform(range=[-15, 15])`).

This is how DALI gives each image its own augmentation while still running a single batched GPU kernel. You can compute arguments yourself too:

```python
shape = fn.peek_image_shape(jpegs)          # [H, W, C] per sample, read from header only
h, w = shape[0], shape[1]
scale = 256.0 / fn.reductions.min(fn.cast(fn.stack(h, w), dtype=types.FLOAT))
images = fn.resize(images, size=fn.stack(h, w) * scale)   # per-sample target sizes
```

Note: CPU operators that produce arguments must run **before** the GPU operators that consume them. Arguments for GPU operators are always CPU `DataNode`s.

## 4. Installation

DALI runs on **Linux with an NVIDIA GPU** (it also works on WSL2). It does not support macOS. If you are on a Mac, use a cloud GPU VM or Google Colab.

```bash
# Pick the wheel that matches your CUDA major version (check with nvidia-smi)
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
# For CUDA 11.x: nvidia-dali-cuda110

pip install torch torchvision   # for the training sections
```

Check the install:

```bash
python -c "import nvidia.dali as dali; print(dali.__version__)"
```

### Dataset layout used in this tutorial

We use the standard "ImageFolder" layout: one subfolder per class.

```
data/
  train/
    cat/  img001.jpg  img002.jpg ...
    dog/  img101.jpg ...
  val/
    cat/ ...
    dog/ ...
```

If you don't have a dataset, a small one such as [Imagenette](https://github.com/fastai/imagenette) works well:

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz   # gives imagenette2-320/train and imagenette2-320/val
```

---

## 5. Step 1: Your first pipeline

```python
# step1_first_pipeline.py
from nvidia.dali import pipeline_def, fn, types

@pipeline_def
def simple_pipeline(data_dir):
    # 1. Read raw (still encoded) JPEG bytes and integer labels from disk (CPU)
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        name="Reader",            # we use this name later for epoch sizing
    )
    # 2. Decode. device="mixed": input on CPU, output on GPU (uses nvJPEG)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    # 3. Resize so the shorter side is 256 px, on the GPU
    images = fn.resize(images, resize_shorter=256)
    return images, labels


pipe = simple_pipeline(
    data_dir="imagenette2-320/train",
    batch_size=8,
    num_threads=4,    # CPU threads for CPU operators
    device_id=0,      # which GPU to use
    seed=42,
)
pipe.build()

images, labels = pipe.run()
print(type(images))                      # TensorListGPU
print("batch size:", len(images))
print("first image shape:", images.as_cpu().at(0).shape)   # (H, W, 3)
print("labels:", [int(labels.at(i)[0]) for i in range(len(labels))])
```

What happens here:

1. **`@pipeline_def`** turns the function into a pipeline factory. The arguments `batch_size`, `num_threads`, `device_id`, and `seed` are added automatically.
2. **`fn.readers.file`** scans the folders and gives each class an integer label in sorted folder order (`cat → 0`, `dog → 1`).
3. **`fn.decoders.image(device="mixed")`** is where most of the speed comes from. Decoding happens on the GPU (and on the A100/H100 hardware JPEG decoder when available).
4. **`pipe.run()`** returns one `TensorList` per output. `images` is on the GPU and `labels` stays on the CPU, because we never moved them.

### Viewing the results

```python
import matplotlib.pyplot as plt

imgs = images.as_cpu()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs.at(i))
    ax.set_title(f"label={int(labels.at(i)[0])}")
    ax.axis("off")
plt.tight_layout(); plt.savefig("step1_batch.png")
```

Notice that the images have **different sizes**. DALI handles batches of different-sized samples without problems. Model training needs a fixed size, so the next step crops the images.

---

## 6. Step 2: A real training augmentation pipeline

This is the standard ImageNet training recipe (random resized crop, horizontal flip, normalize), done entirely on the GPU:

```python
# step2_augmentations.py
from nvidia.dali import pipeline_def, fn, types

IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

@pipeline_def
def train_pipeline(data_dir, crop=224, shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        shard_id=shard_id,          # this process's slice of the dataset
        num_shards=num_shards,      # total number of processes / GPUs
        pad_last_batch=True,
        name="Reader",
    )

    # Decode + random crop in ONE step: only the cropped region is decoded,
    # which saves work.
    images = fn.decoders.image_random_crop(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
        num_attempts=100,
    )
    images = fn.resize(images, resize_x=crop, resize_y=crop)

    # Optional colour jitter, still on the GPU
    images = fn.color_twist(
        images,
        brightness=fn.random.uniform(range=[0.8, 1.2]),
        contrast=fn.random.uniform(range=[0.8, 1.2]),
        saturation=fn.random.uniform(range=[0.8, 1.2]),
    )

    # Flip each sample with 50% probability. fn.random.coin_flip produces a
    # different value for every sample in the batch.
    mirror = fn.random.coin_flip(probability=0.5)

    # One fused kernel: mirror + normalize + HWC→CHW + uint8→float
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        mirror=mirror,
    )
    labels = labels.gpu()           # move labels to the GPU too
    return images, labels


@pipeline_def
def val_pipeline(data_dir, crop=224, size=256, shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="Reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_shorter=size)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),          # center crop
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    return images, labels.gpu()
```

Things to notice:

- **Random parameters are operators too.** `fn.random.uniform(...)` and `fn.random.coin_flip(...)` produce a new value for each sample. You pass them as arguments to other operators. Because of this, each image in the batch gets its own augmentation.
- **Fusing operations matters.** `image_random_crop` combines decode and crop. `crop_mirror_normalize` combines four operations in one GPU kernel. Use fused operators where they exist.
- **Sharding** (`shard_id` / `num_shards`) is all you need for distributed training. Each GPU reads a different part of the dataset.

---

## 7. Step 3: Plugging DALI into PyTorch

`DALIGenericIterator` wraps a pipeline and returns PyTorch tensors that are **already on the GPU**. There is no `.cuda()` call and no host-to-device copy.

```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

pipe = train_pipeline(
    data_dir="imagenette2-320/train",
    batch_size=128, num_threads=8, device_id=0, seed=42,
    prefetch_queue_depth=2,   # prepare the next batches while the model trains
)
pipe.build()

train_loader = DALIGenericIterator(
    pipe,
    output_map=["data", "label"],     # names for the pipeline outputs
    reader_name="Reader",             # lets the iterator know the epoch size
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,                  # start the next epoch automatically
)

for batch in train_loader:
    images = batch[0]["data"]                      # [B, 3, 224, 224] float32, cuda
    labels = batch[0]["label"].squeeze(-1).long()  # [B], cuda
    ...
```

Why `batch[0]`? The iterator can run several pipelines at once (one per GPU in a single process), so it returns a list with one entry per pipeline. With one pipeline, you always use `[0]`.

`LastBatchPolicy` options:
- `FILL`: pads the last batch with repeated samples (all batches have the same size).
- `DROP`: drops the incomplete last batch.
- `PARTIAL`: returns a smaller final batch. **Use this for validation** so every sample is counted exactly once.

---

## 8. Step 4: Full end-to-end training script

This script combines everything so far: DALI pipelines, a torchvision ResNet, mixed precision, and validation. It runs on one GPU or on several with `torchrun`.

```python
# train_dali.py
import argparse, os, time
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STD  = [0.229 * 255, 0.224 * 255, 0.225 * 255]


# ----------------------------- Pipelines ---------------------------------- #
@pipeline_def
def create_pipeline(data_dir, is_training, crop, shard_id, num_shards):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=is_training,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="Reader",
    )
    if is_training:
        images = fn.decoders.image_random_crop(
            jpegs, device="mixed", output_type=types.RGB,
            random_aspect_ratio=[0.75, 4 / 3], random_area=[0.08, 1.0],
            num_attempts=100,
        )
        images = fn.resize(images, resize_x=crop, resize_y=crop)
        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, output_layout="CHW",
            mean=MEAN, std=STD, mirror=mirror,
        )
    else:
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_shorter=int(crop / 0.875))
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, output_layout="CHW",
            crop=(crop, crop), mean=MEAN, std=STD,
        )
    return images, labels.gpu()


def make_loader(data_dir, is_training, batch_size, workers, crop, rank, world, local_rank):
    pipe = create_pipeline(
        data_dir=data_dir, is_training=is_training, crop=crop,
        shard_id=rank, num_shards=world,
        batch_size=batch_size, num_threads=workers,
        device_id=local_rank, seed=12 + rank,
    )
    pipe.build()
    return DALIGenericIterator(
        pipe, ["data", "label"], reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP if is_training else LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )


# ----------------------------- Train / eval ------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total, correct, loss_sum, start = 0, 0, 0.0, time.time()
    for batch in loader:
        x = batch[0]["data"]
        y = batch[0]["label"].squeeze(-1).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    elapsed = time.time() - start
    return loss_sum / total, correct / total, total / elapsed


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = torch.zeros(1, device="cuda")
    total = torch.zeros(1, device="cuda")
    for batch in loader:
        x = batch[0]["data"]
        y = batch[0]["label"].squeeze(-1).long()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        correct += (out.argmax(1) == y).sum()
        total += y.size(0)
    if dist.is_initialized():
        dist.all_reduce(correct); dist.all_reduce(total)
    return (correct / total).item()


# --------------------------------- Main ----------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with train/ and val/")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=10)
    args = p.parse_args()

    # Distributed setup (torchrun sets these environment variables)
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")

    train_loader = make_loader(os.path.join(args.data, "train"), True,
                               args.batch_size, args.workers, args.crop,
                               rank, world, local_rank)
    val_loader = make_loader(os.path.join(args.data, "val"), False,
                             args.batch_size, args.workers, args.crop,
                             rank, world, local_rank)

    model = torchvision.models.resnet50(num_classes=args.num_classes).cuda()
    model = model.to(memory_format=torch.channels_last)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        loss, acc, ips = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        if rank == 0:
            print(f"epoch {epoch+1}/{args.epochs} | loss {loss:.3f} | "
                  f"train acc {acc:.3f} | val acc {val_acc:.3f} | "
                  f"{ips * world:.0f} img/s")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Run it:

```bash
# Single GPU
python train_dali.py --data imagenette2-320 --epochs 5

# 4 GPUs on one machine. Each process reads its own shard.
torchrun --nproc_per_node=4 train_dali.py --data imagenette2-320 --epochs 5
```

### How one batch moves through the system

1. **The reader** (CPU threads) reads file bytes for shard `rank`.
2. **The decoder** (`mixed`) sends the bytes to nvJPEG, and the decoded pixels go straight into GPU memory.
3. **Resize, flip, and normalize** run as CUDA kernels on the same GPU the model uses.
4. **Prefetching**: while the model trains on batch *N*, DALI is already preparing batches *N+1* and *N+2* (`prefetch_queue_depth`).
5. **`DALIGenericIterator`** gives that GPU memory to PyTorch as a `torch.Tensor` without making a copy.
6. **Sharding**: `shard_id=rank, num_shards=world` gives each GPU a separate slice of the data, which `DistributedSampler` would normally do.

---

## 9. Step 5: Custom data with `external_source`

Not every dataset is a folder of JPEGs. Your data might be in a database, in S3, or listed in a CSV with complex labels. `fn.external_source` lets **you** provide the raw data from Python while DALI still does the heavy GPU work.

`train.csv` is a plain CSV with a header and one row per image, with absolute (or working-directory-relative) paths and integer labels:

```
path,label
/data/imagenette2-320/train/n01440764/img_001.JPEG,0
/data/imagenette2-320/train/n02102040/img_002.JPEG,1
```

```python
# step5_external_source.py
import numpy as np
import pandas as pd
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class CSVSource:
    """Gives DALI one sample at a time: (encoded JPEG bytes, label)."""
    def __init__(self, csv_path, shard_id=0, num_shards=1, shuffle=True):
        df = pd.read_csv(csv_path)          # columns: path,label
        self.files = df["path"].tolist()[shard_id::num_shards]
        self.labels = df["label"].tolist()[shard_id::num_shards]
        self.shuffle = shuffle

    def __call__(self, sample_info):
        # sample_info.idx_in_epoch tells you which sample DALI wants
        if sample_info.idx_in_epoch >= len(self.files):
            raise StopIteration                    # end of epoch
        idx = sample_info.idx_in_epoch
        if self.shuffle:
            rng = np.random.default_rng(sample_info.epoch_idx)
            idx = rng.permutation(len(self.files))[idx]
        with open(self.files[idx], "rb") as f:
            encoded = np.frombuffer(f.read(), dtype=np.uint8)
        return encoded, np.array([self.labels[idx]], dtype=np.int64)

    def __len__(self):
        return len(self.files)


@pipeline_def(py_num_workers=4, py_start_method="spawn")
def csv_pipeline(source):
    jpegs, labels = fn.external_source(
        source=source,
        num_outputs=2,
        batch=False,         # source returns ONE sample per call
        parallel=True,       # run the source in py_num_workers processes
        dtype=[types.UINT8, types.INT64],
    )
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(
        images, dtype=types.FLOAT, output_layout="CHW",
        mean=[123.7, 116.3, 103.5], std=[58.4, 57.1, 57.4],
        mirror=fn.random.coin_flip(),
    )
    return images, labels.gpu()


if __name__ == "__main__":
    src = CSVSource("train.csv")
    pipe = csv_pipeline(src, batch_size=64, num_threads=4, device_id=0)
    pipe.build()
    loader = DALIGenericIterator(
        pipe, ["data", "label"], size=len(src),
        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True,
    )
    for batch in loader:
        print(batch[0]["data"].shape, batch[0]["label"].shape)
        break
```

Tips:
- Use `parallel=True` with `batch=False`, and make the source picklable, so DALI can run it in worker processes. This is similar to `DataLoader` workers, but they only handle I/O. Decoding still happens on the GPU.
- `spawn` is required because CUDA cannot be used safely after `fork`.

---

## 10. Step 6: Benchmark DALI against a standard DataLoader

Always measure on your own hardware. This script compares the two **input pipelines only** (no model), which is the cleanest way to see where the time goes.

```python
# step6_benchmark.py
import time, torch, torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from train_dali import create_pipeline   # reuse the pipeline from Step 4

DATA, BS, WORKERS, N_BATCHES = "imagenette2-320/train", 256, 8, 30

def bench(loader, name, get_x):
    it = iter(loader)
    for _ in range(5): next(it)          # warm-up
    torch.cuda.synchronize(); t = time.time()
    for _ in range(N_BATCHES):
        x = get_x(next(it))
    torch.cuda.synchronize()
    print(f"{name:>12}: {N_BATCHES * BS / (time.time() - t):8.0f} img/s")

# --- PyTorch DataLoader (CPU decode + augment) ---
tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dl = DataLoader(ImageFolder(DATA, tf), batch_size=BS, shuffle=True,
                num_workers=WORKERS, pin_memory=True, persistent_workers=True)
bench(dl, "DataLoader", lambda b: b[0].cuda(non_blocking=True))

# --- DALI (GPU decode + augment) ---
pipe = create_pipeline(data_dir=DATA, is_training=True, crop=224, shard_id=0,
                       num_shards=1, batch_size=BS, num_threads=WORKERS, device_id=0)
pipe.build()
dali = DALIGenericIterator(pipe, ["data", "label"], reader_name="Reader", auto_reset=True)
bench(dali, "DALI", lambda b: b[0]["data"])
```

On my machine (L40S, 8 workers, Imagenette, batch 256) this script printed:

```
  DataLoader:     4415 img/s
        DALI:    19197 img/s
```

That is about 4.3× faster. How much faster DALI is depends a lot on your CPU-to-GPU ratio, image resolution, and GPU model. The speedup is largest when there are **few CPU cores per GPU** and **large JPEGs**. With small images and many cores, the difference can be small. That is why you should benchmark before you migrate.

This script is a quick sanity check. Section 13 covers a complete benchmarking method for both images and video, including GPU/CPU utilization, end-to-end training throughput, and a results table you can fill in.

---

## 11. Advanced topics

### 11.1 Performance tuning checklist

Change one thing at a time and measure with the benchmark from Step 6.

| Setting | Starting point | Notes |
|---|---|---|
| `num_threads` | 4 to 8 per GPU | More threads only help if the CPU stage is the bottleneck. Too many threads compete with PyTorch's own threads. |
| `prefetch_queue_depth` | 2 to 3 | Raise it if batch times vary a lot. Each extra batch costs `batch_size × C × H × W × 4` bytes of GPU memory. |
| `hw_decoder_load` | 0.65 to 0.9 | Only matters on GPUs with a hardware JPEG decoder (A100/H100 and newer). |
| `decoders.image_random_crop` | always for training | Decoding only the region you keep can cut decode work by up to 2×. |
| `readers.file(..., dont_use_mmap=True)` | for network filesystems | `mmap` is slow on NFS and Lustre. |
| `read_ahead=True` on the reader | for slow disks | Starts reading the next files earlier. |
| `bytes_per_sample_hint` | your typical sample size | Avoids reallocating buffers again and again. |
| Data format | WebDataset / TFRecord shards | Reading many small files is slow on almost any storage. Large shards read sequentially are much faster. |

**Profile instead of guessing.** DALI marks its work with NVTX ranges, so Nsight Systems shows the DALI stages next to your model's kernels:

```bash
nsys profile -t cuda,nvtx -o dali_profile python train_dali.py --data imagenette2-320 --epochs 1
```

In the timeline, look for **gaps on the GPU** between training steps. If the gaps line up with the DALI CPU stage still running, add threads or move more work to the GPU. If the gaps line up with file reads, the bottleneck is storage.

### 11.2 Conditional execution (if/else inside the pipeline)

Normally a pipeline applies every operator to every sample. With `enable_conditionals=True`, you can write ordinary Python `if` statements on per-sample values. DALI turns them into per-sample branches, so only the selected samples go through each branch:

```python
@pipeline_def(enable_conditionals=True)
def conditional_aug(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed")

    # About 30% of samples get rotated; the rest get a colour jitter instead
    if fn.random.coin_flip(probability=0.3):
        images = fn.rotate(images, angle=fn.random.uniform(range=[-30, 30]), fill_value=0)
    else:
        images = fn.hsv(images, hue=fn.random.uniform(range=[-20, 20]))

    images = fn.resize(images, resize_x=224, resize_y=224)
    return images, labels.gpu()
```

This is how you build RandAugment / AutoAugment-style policies. DALI also ships these ready to use:

```python
from nvidia.dali.auto_aug import rand_augment, auto_augment, trivial_augment

@pipeline_def(enable_conditionals=True)
def randaug_pipeline(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = rand_augment.rand_augment(images, n=2, m=9)      # on the GPU
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout="CHW",
                                      mean=[123.7, 116.3, 103.5], std=[58.4, 57.1, 57.4])
    return images, labels.gpu()
```

### 11.3 Custom operations: three options, from easiest to fastest

1. **`fn.python_function`**: any Python/NumPy code. It's easy to write, but it holds the GIL and runs one batch at a time, so use it for prototyping.
   ```python
   def add_noise(img):
       return (img + np.random.normal(0, 5, img.shape)).clip(0, 255).astype(np.uint8)
   images = fn.python_function(images, function=add_noise)  # CPU images only
   ```
2. **`dali_torch.fn.torch_python_function`**: write the operation with PyTorch tensors, and it can run on the GPU.
   ```python
   import nvidia.dali.plugin.pytorch as dali_torch
   images = dali_torch.fn.torch_python_function(
       images, function=lambda t: torch.clamp(t + torch.randn_like(t.float()) * 5, 0, 255).byte(),
       batch_processing=False, device="gpu")
   ```
3. **Numba (`nvidia.dali.plugin.numba.fn.experimental.numba_function`)** or a **C++/CUDA custom operator** (built as a `.so` and loaded with `nvidia.dali.plugins.load_library`). Use these when the operation sits on the critical path and you need full speed.

### 11.4 Debugging pipelines

Because the pipeline is a graph, you can't normally `print()` intermediate values. There are two ways around that:

```python
# 1. Debug mode: runs eagerly, so you can print and inspect intermediate results
#    (the debug= flag lives in the experimental pipeline_def)
from nvidia.dali.pipeline.experimental import pipeline_def as debug_pipeline_def

@debug_pipeline_def(debug=True)
def dbg(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir)
    images = fn.decoders.image(jpegs, device="cpu")
    print(images.shape())          # real shapes for this batch
    return images, labels

# 2. Return extra outputs temporarily, then look at them
#    return images, labels, fn.peek_image_shape(jpegs)
```

Debug mode is much slower, so use it only to find bugs.

### 11.5 Checkpointing: resume mid-epoch with the same data order

For long training runs you want a restart to continue with the **same** shuffle order and random augmentation state. DALI can save and restore its readers and random generators:

```python
@pipeline_def(enable_checkpointing=True)
def ckpt_pipeline(data_dir):
    ...

loader = DALIGenericIterator(pipe, ["data", "label"], reader_name="Reader")
# ... train some steps ...
dali_state = loader.checkpoints()             # list with one string per pipeline
torch.save({"model": model.state_dict(), "dali": dali_state}, "ckpt.pt")

# On restart:
state = torch.load("ckpt.pt")
pipe = ckpt_pipeline(data_dir=..., batch_size=..., num_threads=..., device_id=0,
                     checkpoint=state["dali"][0])
```

### 11.6 GPUDirect Storage: skip the CPU completely

For data that is already decoded (NumPy arrays, medical volumes, or pre-processed tensors), `fn.readers.numpy(device="gpu")` uses **GPUDirect Storage**. Data goes from NVMe straight into GPU memory and never passes through a CPU bounce buffer:

```python
@pipeline_def
def npy_pipeline(root):
    volumes = fn.readers.numpy(device="gpu", file_root=root, file_filter="*.npy",
                               random_shuffle=True, name="Reader")
    # Random 3D crop, still on the GPU (the 128³ crop size is just an example)
    volumes = fn.crop(volumes, crop=(128, 128, 128),
                      crop_pos_x=fn.random.uniform(range=[0, 1]),
                      crop_pos_y=fn.random.uniform(range=[0, 1]),
                      crop_pos_z=fn.random.uniform(range=[0, 1]))
    return volumes
```

This needs the GDS / `cuFile` stack and a supported filesystem. Without them, DALI falls back to a slower compatibility mode.

### 11.7 Reusing the same pipeline for inference (Triton)

A common source of bugs is preprocessing that differs slightly between training and serving. DALI lets you **serialize** the validation pipeline and run exactly the same graph in Triton Inference Server with the DALI backend:

```python
@pipeline_def(batch_size=64, num_threads=4, device_id=0)
def serving_pipe():
    jpegs = fn.external_source(name="INPUT", dtype=types.UINT8)  # bytes sent by clients
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.resize(images, resize_shorter=256)
    return fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout="CHW",
                                    crop=(224, 224), mean=MEAN, std=STD)

serving_pipe().serialize(filename="model_repository/preprocess/1/model.dali")
```

In Triton, combine this with your model in an *ensemble*. Clients send raw JPEG bytes, and decoding, preprocessing and inference all run on the server GPU.

## 12. Step 7: Ingesting and processing video

Video is where DALI helps the most. Decoding video on the CPU is very slow: one 1080p clip can take a full CPU core. A typical PyTorch video loader (with PyAV or decord) decodes **every** frame on the CPU, converts it to RGB, and then throws most of the frames away. DALI uses the GPU's **NVDEC** hardware decoder, which is separate from the CUDA cores, so decoding barely slows down training.

### 12.1 How the video reader works

```
 .mp4 file ──demux (CPU)──► compressed packets ──NVDEC (GPU)──► frames in GPU memory
                                                           │
                         seek to nearest keyframe ◄────────┘ (only decode what's needed)
```

- `fn.readers.video` splits each video into **sequences** (clips). A clip is `sequence_length` frames, taken every `step` frames, with `stride` frames between the frames inside a clip.
- To read a clip, it **seeks to the nearest keyframe** before the clip and decodes forward from there. Keyframe spacing (the GOP size) therefore has a direct effect on speed.
- Output layout is **`FHWC`** (frames, height, width, channels), as `uint8` on the GPU.
- Each clip counts as one "sample", so `batch_size=8` means 8 clips of `sequence_length` frames each.

```
video frames:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
sequence_length=4, stride=2, step=8
clip 0:        0   2   4   6
clip 1:                        8    10    12    14
clip 2:                                             16 ...
```

### 12.2 Prepare the videos (important)

NVDEC is particular about its input. Before training, re-encode your dataset once to:
- **H.264 or HEVC** (VP9 and AV1 work on newer GPUs),
- a **constant frame rate** (variable-frame-rate phone videos cause seeking errors and wrong frames),
- **short keyframe intervals**, so random access is cheap,
- a **moderate resolution**. Decoding 4K just to resize to 112 px wastes most of the work.

```bash
# Re-encode every video: 30 fps CFR, 256 px short side, keyframe every 16 frames, no audio
mkdir -p videos_cfr
for f in videos/*/*.mp4; do
  out="videos_cfr/${f#videos/}"; mkdir -p "$(dirname "$out")"
  ffmpeg -loglevel error -y -i "$f" \
    -vf "fps=30,scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'" \
    -c:v libx264 -preset fast -crf 23 -g 16 -keyint_min 16 -sc_threshold 0 \
    -an "$out"
done
```

The folder layout is the same as for images: one subfolder per class (for example UCF101 or Kinetics):

```
videos_cfr/
  train/
    archery/       v_001.mp4 ...
    basketball/    v_101.mp4 ...
  val/ ...
```

### 12.3 Training pipeline for action recognition

```python
# step8_video.py
import os, sys, torch, torch.nn as nn, torchvision
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# Kinetics normalization constants (used by torchvision video models), scaled to 0-255
MEAN = [0.43216 * 255, 0.394666 * 255, 0.37645 * 255]
STD  = [0.22803 * 255, 0.22145 * 255, 0.216989 * 255]

@pipeline_def
def video_train_pipeline(video_root, seq_len=16, crop=112, shard_id=0, num_shards=1):
    # file_root + class subfolders -> automatic integer labels, just like readers.file
    frames, labels = fn.readers.video(
        device="gpu",
        file_root=video_root,
        sequence_length=seq_len,
        stride=2,                  # temporal subsampling: every 2nd frame (~0.5 s @ 30 fps x 16)
        step=seq_len * 2,          # don't overlap clips inside one video
        random_shuffle=True,
        initial_fill=64,           # shuffle buffer size, in clips
        shard_id=shard_id, num_shards=num_shards,
        pad_last_batch=True,
        skip_vfr_check=False,      # fail loudly on variable-frame-rate files
        name="Reader",
    )
    # frames: [F, H, W, C] uint8 on GPU, per clip

    # Spatial augmentation. Random parameters are drawn ONCE PER CLIP, so every
    # frame in a clip gets the same crop and flip and the motion stays consistent.
    frames = fn.resize(frames, resize_shorter=128)
    frames = fn.crop_mirror_normalize(
        frames,
        crop=(crop, crop),
        crop_pos_x=fn.random.uniform(range=[0.0, 1.0]),
        crop_pos_y=fn.random.uniform(range=[0.0, 1.0]),
        mirror=fn.random.coin_flip(probability=0.5),
        mean=MEAN, std=STD,
        dtype=types.FLOAT,
        output_layout="CFHW",      # what torchvision's 3D CNNs expect: [C, T, H, W]
    )
    return frames, labels


@pipeline_def
def video_val_pipeline(video_root, seq_len=16, crop=112, shard_id=0, num_shards=1):
    frames, labels = fn.readers.video(
        device="gpu", file_root=video_root, sequence_length=seq_len,
        stride=2, step=seq_len * 2, random_shuffle=False,
        shard_id=shard_id, num_shards=num_shards, pad_last_batch=True, name="Reader",
    )
    frames = fn.resize(frames, resize_shorter=128)
    frames = fn.crop_mirror_normalize(
        frames, crop=(crop, crop), mean=MEAN, std=STD,
        dtype=types.FLOAT, output_layout="CFHW",
    )
    return frames, labels


def make_loader(pipe_fn, root, batch_size, policy):
    pipe = pipe_fn(video_root=root, batch_size=batch_size, num_threads=4,
                   device_id=0, seed=42)
    pipe.build()
    return DALIGenericIterator(pipe, ["video", "label"], reader_name="Reader",
                               last_batch_policy=policy, auto_reset=True)


def main(root, num_classes, epochs=3, batch_size=8):
    train = make_loader(video_train_pipeline, os.path.join(root, "train"),
                        batch_size, LastBatchPolicy.DROP)
    val = make_loader(video_val_pipeline, os.path.join(root, "val"),
                      batch_size, LastBatchPolicy.PARTIAL)

    # R(2+1)D-18: a small 3D CNN that takes [B, 3, T, H, W]
    model = torchvision.models.video.r2plus1d_18(num_classes=num_classes).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train):
            x = batch[0]["video"]                       # [B, 3, 16, 112, 112] cuda float
            y = batch[0]["label"].squeeze(-1).long().cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if i % 20 == 0:
                print(f"epoch {epoch} iter {i} loss {loss.item():.3f}")

        model.eval(); correct = total = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for batch in val:
                y = batch[0]["label"].squeeze(-1).long().cuda()
                correct += (model(batch[0]["video"]).argmax(1) == y).sum().item()
                total += y.numel()
        print(f"epoch {epoch}: clip-level val acc {correct / total:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], num_classes=int(sys.argv[2]))
```

```bash
python step8_video.py videos_cfr 101     # e.g. UCF101
```

A few points to note:
- **Labels stay on the CPU.** With `device="gpu"`, the reader puts frames on the GPU but returns labels as CPU tensors, so the training loop calls `.cuda()` on them.
- **Same augmentation for every frame in a clip.** In image pipelines, each sample gets different random values. Here the sample *is* the clip, so the crop and flip are shared by all of its frames, which is what you want for video. If you need a separate value per frame, use `fn.per_frame(...)` on the argument.
- **`output_layout="CFHW"`** does the transpose inside the same fused kernel, so there's no `permute()` in PyTorch.
- **Clip vs video accuracy.** The validation loop above scores single clips. Published numbers usually average predictions over several clips per video; section 12.5 shows how to keep track of which clip came from which video.

### 12.4 Explicit file lists, labels and time ranges

Real datasets usually label **time segments**, not whole files. In that case, use a `file_list` text file with lines of `path label start end`:

```
# train_list.txt  (start/end in seconds by default; use file_list_frame_num=True for frame numbers)
/data/v/match01.mp4 3 12.0 18.5
/data/v/match01.mp4 7 40.2 44.0
/data/v/match02.mp4 3 0.0 6.0
```

```python
frames, labels = fn.readers.video(
    device="gpu", file_list="train_list.txt",
    sequence_length=16, stride=2, random_shuffle=True, name="Reader",
)
```

DALI only takes clips from inside each `[start, end)` range. This lets you train on labelled segments of long videos without cutting the files.

### 12.5 Batch inference over long videos (frame extraction / embeddings)

Another common job is going through every frame of a long video, for example to compute per-second embeddings for search, detect objects, or create thumbnails. Read the clips in order, without overlap, and ask DALI for **frame numbers** and **timestamps** so you can map each output back to a time in the video:

```python
@pipeline_def
def video_scan_pipeline(filenames, seq_len=8):
    frames, labels, frame_num, timestamps = fn.readers.video(
        device="gpu",
        filenames=filenames,
        labels=list(range(len(filenames))),   # use the label as a "video id"
        sequence_length=seq_len,
        step=seq_len,               # back-to-back clips, no gaps or overlap
        stride=1,
        random_shuffle=False,
        pad_sequences=True,         # pad the last short clip instead of dropping it
        enable_frame_num=True,      # index of the first frame in the clip
        enable_timestamps=True,     # timestamp (s) of each frame
        name="Reader",
    )
    # Treat every frame as its own image: [F,H,W,C] -> resize -> normalize per frame
    frames = fn.resize(frames, resize_x=224, resize_y=224)
    frames = fn.crop_mirror_normalize(
        frames, dtype=types.FLOAT, output_layout="FCHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return frames, labels, frame_num, timestamps


def embed_videos(filenames, batch_size=4, seq_len=8):
    pipe = video_scan_pipeline(filenames=filenames, seq_len=seq_len,
                               batch_size=batch_size, num_threads=4, device_id=0)
    pipe.build()
    it = DALIGenericIterator(pipe, ["frames", "vid", "frame_num", "ts"],
                             reader_name="Reader",
                             last_batch_policy=LastBatchPolicy.PARTIAL)

    backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()                    # 2048-d embeddings
    backbone = backbone.cuda().eval()

    results = []                                   # (video_id, timestamp, embedding)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in it:
            b = batch[0]
            B, F = b["frames"].shape[:2]
            emb = backbone(b["frames"].flatten(0, 1))           # [B*F, 2048]
            emb = emb.view(B, F, -1).float().cpu()
            for i in range(B):
                vid = int(b["vid"][i])
                for f in range(F):
                    ts = float(b["ts"][i][f])
                    if f > 0 and ts == 0.0:                     # skip padded frames
                        continue
                    results.append((filenames[vid], ts, emb[i, f]))
    return results
```

Because DALI decodes clips that come one after another, NVDEC reads each video almost sequentially, and seeking costs very little. On a modern data-center GPU this is clearly faster than decoding with FFmpeg on the CPU: in my test (section 13.6, L40S, 8 cores) data loading was 1.3–1.9× faster and R(2+1)D training was 1.8× faster. Benchmark it on your own videos to get real numbers.

### 12.6 Decoding videos you already have in memory

If the videos come from object storage, a message queue, or a web upload instead of local files, combine `external_source` with the **video decoder** (`fn.experimental.decoders.video`, available in recent DALI releases):

```python
@pipeline_def(batch_size=2, num_threads=4, device_id=0)
def in_memory_video():
    encoded = fn.external_source(name="VIDEO_BYTES", dtype=types.UINT8, ndim=1)
    frames = fn.experimental.decoders.video(encoded, device="mixed")   # all frames, FHWC
    frames = fn.resize(frames, resize_x=224, resize_y=224)
    return frames

pipe = in_memory_video(); pipe.build()
raw = [np.fromfile(p, dtype=np.uint8) for p in ["a.mp4", "b.mp4"]]
pipe.feed_input("VIDEO_BYTES", raw)
(frames,) = pipe.run()
```

This decodes **whole** videos, so keep them short (for example clips that were already cut), or use it inside Triton for video inference requests.

### 12.7 Video gotchas

| Symptom | Fix |
|---|---|
| `Variable frame rate videos are not supported` / wrong frames | Re-encode to a constant frame rate (`-vf fps=30`), as in 12.2. |
| Very slow random access | The keyframe interval is too large. Re-encode with `-g 16` (or similar). |
| `Unsupported codec` | Re-encode to H.264. Check which codecs your GPU's NVDEC supports in NVIDIA's support matrix. |
| GPU memory spikes | Lower the resolution before training. Each decoded 1080p frame is about 6 MB, so 16 frames × batch 8 ≈ 800 MB before resize. |
| Videos shorter than one clip get skipped | Lower `sequence_length`/`stride`, or use `pad_sequences=True`. |
| Epoch size isn't "number of videos" | Expected: one sample = one clip, and a video produces many clips. Use `reader_name` to get the right epoch length. |

## 13. Step 8: Benchmarking properly (images and video)

Claims like "DALI is 3× faster" (or the numbers in my own TL;DR) don't mean much until you've measured it on **your** data, **your** GPU, and **your** CPU budget. This section gives you a repeatable method and a single script that produces a results table.

### 13.1 What to measure

| Metric | Why it matters | How |
|---|---|---|
| **Loader-only throughput** (samples/s) | The fastest rate the input pipeline can deliver. It's an upper bound on training speed. | Iterate the loader and do nothing with the batch except move it to the GPU. |
| **End-to-end throughput** (samples/s) | What you actually get when training. This is the number that matters. | Run real forward/backward/optimizer steps. |
| **GPU utilization** (%) | A low or fluctuating value means the GPU is waiting for data. | Sample NVML every 100 ms while running. |
| **CPU utilization** (%) | How much CPU the loader uses. On shared nodes, freeing CPU is a benefit in itself. | Sample `psutil.cpu_percent`. |
| **Peak GPU memory** (MB) | DALI uses GPU memory for decoding and prefetching. Make sure your model still fits. | NVML memory info (includes DALI's pools, unlike `torch.cuda.max_memory_allocated`). |

Compare the two results: if **loader-only throughput is much higher than end-to-end throughput** for both loaders, the model is the bottleneck and DALI won't help much. If the PyTorch loader-only number is **lower** than what the model could consume, switching to DALI will speed up training.

### 13.2 Rules for a fair benchmark

1. **Same work on both sides.** Use the same augmentations, output size, dtype and batch size, and include the transfer to the GPU for the PyTorch loader.
2. **Warm up first.** Skip the first iterations (worker startup, cuDNN autotuning, DALI pools growing).
3. **Call `torch.cuda.synchronize()`** before reading the timer. CUDA is asynchronous, so without it you measure how fast work was *queued*, not how fast it ran.
4. **Repeat** (3 or more runs) and report the median.
5. **Control the file cache.** The second run reads from RAM. Either always warm the cache (simplest), or drop it before every run to measure cold-disk performance:
   `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`
6. **Simulate the CPU budget of your real setup.** On an 8-GPU node with 64 cores, each GPU effectively gets about 8 cores. Benchmarking one GPU with all 64 cores makes the CPU loader look better than it will be in practice. Limit the cores with `taskset`:
   `taskset -c 0-7 python benchmark.py ...`
7. **Sweep the worker count.** A loader might only look slow because it has too few workers. Test 4, 8, and 16 (and more if you have the cores).

### 13.3 The benchmark script

It reuses the pipelines from earlier steps (`train_dali.py` and `step8_video.py`), so it measures exactly the pipelines you train with.

```bash
pip install nvidia-ml-py psutil av pandas matplotlib
```

```python
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
```

### 13.4 Plot the results

```python
# plot_results.py
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
for (modality, mode), g in df.groupby(["modality", "mode"]):
    pivot = g.pivot_table(index="workers", columns="loader", values="samples_per_s")
    ax = pivot.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylabel("clips / s" if modality == "video" else "images / s")
    ax.set_title(f"{modality} – {mode}  ({g['gpu'].iloc[0]})")
    plt.tight_layout()
    plt.savefig(f"bench_{modality}_{mode}.png", dpi=150)
    print("wrote", f"bench_{modality}_{mode}.png")
```

### 13.5 A full benchmark run

```bash
# 1. Images, input pipeline only, sweeping workers, limited to 8 cores (one GPU's share)
taskset -c 0-7 python benchmark.py image --data imagenette2-320/train --workers 2 4 8 --batch-size 128

# 2. Images, end-to-end with ResNet-50 training steps
taskset -c 0-7 python benchmark.py image --data imagenette2-320/train --workers 8 --batch-size 128 --e2e

# 3. Video, input pipeline only, and end-to-end with R(2+1)D-18
taskset -c 0-7 python benchmark.py video --data videos_cfr/train --workers 4 8 --batch-size 8
taskset -c 0-7 python benchmark.py video --data videos_cfr/train --workers 8 --batch-size 8 --e2e

# 4. Charts
python plot_results.py

# 5. Profile one configuration to see WHY the numbers look the way they do
nsys profile -t cuda,nvtx,osrt -o dali_e2e \
  python benchmark.py image --data imagenette2-320/train --workers 8 --e2e --repeats 1
```

### 13.6 Results

These are the numbers from running section 13.5 on my machine. Yours will differ, so fill in the same table for your own hardware.

**Setup:** GPU: NVIDIA L40S (46 GB) · CPU: AMD EPYC 9354, restricted to 8 cores with `taskset -c 0-7` · Storage: local disk, warm page cache · DALI 2.4.0 · PyTorch 2.8.0 (CUDA 12.8) · Image data: Imagenette2-320 train (9,469 JPEGs, batch 128) · Video data: 42 clips of 20 s each (about 25,000 frames), cut from three open-licensed Blender films (*Big Buck Bunny*, *Sintel*, *Tears of Steel*) that I downloaded from YouTube at 1080p H.264 (24 and 60 fps sources, 2 aspect ratios) with `yt-dlp`, then re-encoded as in section 12.2 (30 fps, 256 px short side, keyframe every 16 frames), 16-frame clips, batch 8. This mimics a real ingestion pipeline: download, normalize, train. Each number is the mean ± std over 3 repeats.

**Images, data loading only** (no model):

| Workers | PyTorch (img/s) | DALI (img/s) | Speedup |
|---|---|---|---|
| 2 | 952 ± 10 | 17,781 ± 386 | 18.7× |
| 4 | 1,709 ± 66 | 18,964 ± 150 | 11.1× |
| 8 | 2,805 ± 879 | 19,086 ± 454 | 6.8× |

**Everything together:**

| Workload | Loader | Workers | Throughput | GPU util | CPU util | Peak GPU mem | Speedup |
|---|---|---|---|---|---|---|---|
| Image, loader-only | PyTorch | 8 | 2,805 img/s | 6% | 12% | 1.6 GB | 1.0× |
| Image, loader-only | DALI | 8 | 19,086 img/s | 24% | 10% | 1.7 GB | 6.8× |
| Image, ResNet-50 e2e | PyTorch | 8 | 1,110 img/s | 92% | 10% | 7.8 GB | 1.0× |
| Image, ResNet-50 e2e | DALI | 8 | 1,204 img/s | 98% | 8% | 8.2 GB | 1.08× |
| Video, loader-only | PyTorch (PyAV) | 8 | 96 clips/s | 1% | 12% | 1.4 GB | 1.0× |
| Video, loader-only | DALI (NVDEC) | 8 | 126 clips/s | 2% | 7% | 1.4 GB | 1.3× |
| Video, R(2+1)D e2e | PyTorch (PyAV) | 8 | 62 clips/s | 40% | 11% | 4.0 GB | 1.0× |
| Video, R(2+1)D e2e | DALI (NVDEC) | 8 | 109 clips/s | 72% | 15% | 4.3 GB | 1.8× |

**What these numbers say** (using the table in section 13.7):

- **Images, loader-only:** DALI delivers about 19,000 img/s no matter how many workers it gets. The PyTorch loader grows with workers (952 → 2,805 img/s) but is still 6.8× slower at 8 workers, and its numbers vary a lot between runs at 8 workers (± 879).
- **Images, end-to-end:** DALI is only 8% faster. DALI could feed about 19,000 img/s, but ResNet-50 training tops out near 1,200 img/s, so the model is the bottleneck (DALI GPU util 98%). The PyTorch run is already at 92% GPU utilization, so on this machine with 8 cores the CPU loader was almost keeping up. A slower CPU, fewer cores, or a lighter model would show a larger gain.
- **Video:** DALI helps most in training: 1.8× in end-to-end throughput, with GPU utilization going from 40% to 72%. In loader-only mode the PyTorch (PyAV) loader scales with workers (65 clips/s at 4, 96 at 8), while DALI stays at about 126 clips/s from 4 workers on, so DALI's advantage shrinks from 1.9× to 1.3× when the CPU has more cores to spend.

**Caveats, so you don't over-read this:**

- One GPU model, one CPU, one dataset. Imagenette images are small (320 px); full-size ImageNet JPEGs are about 10× larger, which usually favors DALI more.
- The video set is small (42 clips, 3 films, all animation) and sits in the page cache. Real footage (more motion, more noise) is harder to decode, so treat the video numbers as relative, not absolute.
- **DALI is not hardware-limited in the video loader-only test.** While it ran at 126 clips/s I sampled the GPU with `nvidia-smi dmon -s u`: the hardware video decoder (NVDEC) was about 32% busy and the GPU's compute units about 2%. So the plateau comes from something else, such as how the reader seeks and schedules clips, not from a full GPU. I did not dig further.
- The GPU utilization column is sampled by NVML in a background thread. For loader-only runs it is low because the GPU is mostly idle, not because DALI is idle.
- I did not tune either loader beyond the settings in the scripts.

### 13.7 How to read the results

| What you see | What it means | What to do |
|---|---|---|
| PyTorch e2e ≪ DALI e2e, PyTorch GPU util low | The CPU was the bottleneck, and DALI fixed it. | Switch to DALI. 🎉 |
| PyTorch e2e ≈ DALI e2e, both GPU util ~100% | The model is the bottleneck. | DALI only frees up CPU. Keep it if that CPU is useful elsewhere; otherwise it's optional. |
| PyTorch improves a lot as workers go up | The CPU loader was only under-provisioned. | Check whether you'll actually have those cores per GPU in production (multi-GPU, shared nodes). |
| DALI loader-only is low too | Storage or file reading is the bottleneck. | Check `iostat -x 1`. Use local NVMe, WebDataset/TFRecord shards, GPUDirect Storage, or a warm page cache. |
| DALI e2e is slightly *lower* than PyTorch | The DALI GPU work competes with the model for the GPU, or the model barely fits in memory. | Move some augmentations to the CPU (decode on GPU, augment on CPU), lower `prefetch_queue_depth`, or raise `hw_decoder_load` so the hardware decoder does more of the work. |
| Video: DALI far ahead even with many workers | Decoding video on the CPU costs far more than decoding JPEGs. | Expected. NVDEC is dedicated hardware. |
| Video: DALI throughput drops on some datasets | Large keyframe intervals or high resolution make each seek expensive. | Re-encode the videos (section 12.2). |

### 13.8 Decoding every frame: CPU vs FFmpeg-GPU vs DALI (1080p and 720p)

The benchmarks above sample *clips* (16 frames, every second frame). Many video jobs instead need **every frame**: frame extraction, embeddings, transcoding, labeling. For that, the question is raw decode throughput, with no resize, crop or model. `decode_all_frames.py` decodes every frame of every video three ways and reports frames per second:

- **CPU:** FFmpeg software decoding, 8 processes with 1 thread each (8 cores in total).
- **FFmpeg GPU:** the same 8 processes with `-hwaccel cuda -hwaccel_output_format cuda`, so decoding runs on NVDEC and the frames stay in GPU memory.
- **DALI:** `fn.readers.video(device="gpu", stride=1, step=sequence_length)`, which also uses NVDEC and returns consecutive, non-overlapping sequences, so each frame is delivered once and stays on the GPU.

```bash
taskset -c 0-7 python decode_all_frames.py --data videos_1080/train --label 1080p --out decode_results.csv
taskset -c 0-7 python decode_all_frames.py --data videos_720/train  --label 720p  --out decode_results.csv
python plot_decode.py decode_results.csv images/
```

<details><summary>decode_all_frames.py</summary>

```python
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
```

</details>

**Setup:** same machine as section 13.6 (L40S, 8 cores, FFmpeg 6.1.1, DALI 2.4.0). Both sets are H.264 with a keyframe every 48 frames and contain the same 9 clips of 60 s (19,457 frames), cut from the YouTube downloads of the three Blender films (24 and 60 fps):

- **1080p:** 1920 px wide (1920×1080, 1920×818 and 1920×800, depending on the film's aspect ratio).
- **720p:** the same clips scaled to 1280 px wide (1280×720, 1280×546 and 1280×534).

Median of 3 repeats after a warm-up pass; frames/s counts frames actually decoded.

![Frames decoded per second for CPU FFmpeg, FFmpeg + NVDEC and DALI at 1080p and 720p](images/decode_compare.png)

| Method | 1080p frames/s | 720p frames/s |
|---|---|---|
| CPU FFmpeg (8 cores) | 1,796 | **3,902** |
| FFmpeg + NVDEC | **2,346** | 3,686 |
| DALI (256-frame sequences) | 882 | 1,715 |

NVDEC utilization during these runs: FFmpeg + NVDEC 80% (1080p) and 65% (720p); DALI 33% at both resolutions.

![DALI decode speed against sequence length, compared with the two FFmpeg baselines](images/decode_dali_seq.png)

| DALI sequence length | 1080p frames/s | 720p frames/s |
|---|---|---|
| 16 | 382 | 734 |
| 64 | 702 | 1,352 |
| 128 | 812 | 1,569 |
| 256 | 882 | 1,715 |

**What this shows:**

1. **There is a crossover around 1080p.** At 1080p, NVDEC through FFmpeg is the fastest (1.3× the 8-core CPU). At 720p the CPU is slightly ahead (3,902 vs 3,686 fps), because each frame is cheap for the CPU while the GPU path still pays per-frame costs. As resolution drops further, CPU wins by more. I did not test 4K, where the gap should widen in the GPU's favor.
2. **For full decoding, DALI was not the fastest at either resolution.** It reaches 38–47% of the FFmpeg + NVDEC speed. The DALI reader is built to *sample clips* for training (shuffling, strides, sharding), which is where it won in section 13.6. It works as a plain "decode everything" tool but is not optimized for it.
3. **DALI's speed depends on sequence length.** It re-seeks to a keyframe for each sequence, so short sequences waste work: with a keyframe every 48 frames, going from 16-frame to 256-frame sequences more than doubles throughput at both resolutions (2.3× at 1080p, 2.3× at 720p). If you must decode everything with DALI, use long sequences. Batch size made no difference in my earlier 1080p tests (1, 8 and 32 gave the same result).
4. **DALI used only about a third of the decoder.** NVDEC stayed at 33% in every DALI run, while FFmpeg's 8 parallel processes reached 80% at 1080p. The L40S has several NVDEC engines, and this is consistent with the DALI reader keeping only one of them busy. I did not confirm that, so treat it as a hypothesis.
5. **Rule of thumb:** high-resolution video and many frames → GPU decoding pays off. At 720p or below, with plenty of CPU cores, plain CPU FFmpeg is hard to beat. Measure with your own files.

**Why is DALI slower than the CPU here?**

This is a fair question, because "GPU decoding" sounds like it should win. Here is what I measured and what I think explains it.

*Measured:*

- DALI's video reader decoded 882 frames/s at 1080p. The same GPU, driven by 8 FFmpeg processes, decoded 2,346 frames/s. So the GPU is not the limit; the way DALI's reader uses it is.
- NVDEC (the GPU's hardware video decoder) was 33% busy in every DALI run and 65–80% busy in the FFmpeg runs.
- DALI got faster as the sequence length grew (382 → 882 frames/s at 1080p, 734 → 1,715 at 720p), and batch size made no difference.

*Likely reasons (inferred from the numbers above, not proven):*

1. **The CPU baseline is strong.** It uses 8 modern CPU cores doing nothing but decoding, with an optimized software decoder: about 225 frames/s per core at 1080p. A fixed amount of GPU decoder hardware has to beat that.
2. **DALI's reader appears to use one decoder engine.** The L40S has more than one NVDEC engine, and parallel FFmpeg processes spread across them. DALI's constant 33% fits a single busy engine. If the card has 3, FFmpeg's 2,346 frames/s is about 780 per engine, close to DALI's 882. I did not confirm the engine count or that DALI uses only one.
3. **DALI's reader is designed for sampling clips, not for streaming a whole file.** For each sequence it seeks back to a keyframe and decodes forward. With a keyframe every 48 frames, short sequences decode frames that are then discarded, which is why longer sequences help. Training needs random clips from many videos, which is the case it is built and optimized for.
4. **Both GPU paths pay fixed costs that the CPU does not.** Frames are copied or kept in GPU memory, and CUDA has to start up. This matters more at low resolution, where each frame is cheap for the CPU. That is why the CPU pulls ahead at 720p.

*What this does not show:* that DALI is slow in general. The test counted only decode speed. It ignores CPU usage, where the CPU path used all 8 cores and DALI needed very few. That is the main reason to use DALI in training: it frees the CPU for other work. I did not run a test with fewer CPU cores, where I would expect the CPU path to slow down and DALI to stay about the same. For the clip-sampling workload DALI's reader was built for, it was 1.3–1.9× faster than the PyAV loader (section 13.6).

**Bottom line:** if the job is "decode every frame as fast as possible", use FFmpeg with NVDEC at 1080p and above, or plain CPU FFmpeg at 720p and below, when you have spare cores. If the job is "feed random clips to a model while keeping the CPU free", DALI is the better tool.

**Caveats:**

- One GPU (L40S), one CPU (EPYC 9354, 8 cores), H.264 only, animation rather than camera footage. HEVC, AV1 or noisy footage will behave differently, and fewer CPU cores would favor the GPU more.
- Small set: 9 clips per resolution. In earlier runs on a set of 256p clips, FFmpeg with `-hwaccel cuda` was dominated by process startup (0.62 s for one 600-frame clip versus 0.25 s on the CPU, mostly CUDA initialization). The same effect, to a smaller degree, is included in the FFmpeg + NVDEC numbers here, because it starts one process per file.
- DALI delivers whole sequences, so it drops up to `sequence_length - 1` frames at the end of each video (18,432 of 19,457 frames for 256-frame sequences, 5% fewer than FFmpeg decodes). The speeds above count only frames actually delivered.
- FFmpeg here discards frames (`-f null`). A real job also has to resize, copy or encode them, which costs both paths extra.
- **Gotcha:** my first 1080p files were cut with `ffmpeg -c copy` at non-keyframe boundaries. FFmpeg decoded them (with `mmco: unref short failure` warnings) but DALI's reader aborted with `Current pipeline object is no longer valid`. Re-encode the clips (section 12.2) instead of stream-copying.

## 14. Beyond images and video (short overview)

DALI also supports other data types. The overall pattern stays the same: reader → decode → augment → iterator.

```python
# Audio: decode + mel spectrogram on the GPU
audio, sr = fn.decoders.audio(encoded, dtype=types.FLOAT, downmix=True)
spec = fn.spectrogram(audio.gpu(), nfft=512, window_length=400, window_step=160)
mel  = fn.mel_filter_bank(spec, sample_rate=16000, nfilter=80)

# Object detection: crop + flip boxes together with the image
jpegs, bboxes, labels = fn.readers.coco(file_root=img_dir, annotations_file=ann,
                                        ltrb=True, name="Reader")
crop_begin, crop_size, bboxes, labels = fn.random_bbox_crop(
    bboxes, labels, bbox_layout="xyXY", allow_no_crop=True)
images = fn.decoders.image_slice(jpegs, crop_begin, crop_size, device="mixed")
```

Other readers you may need: `fn.readers.tfrecord`, `fn.readers.webdataset`, `fn.readers.numpy` (which supports GPUDirect Storage), and `fn.readers.caffe` / `mxnet`.

---

## 15. Common problems and fixes

| Symptom | Cause / fix |
|---|---|
| `TypeError: Pipeline.__init__() got an unexpected keyword argument 'debug'` | In DALI 2.x, `debug=True` is only accepted by the experimental decorator: `from nvidia.dali.pipeline.experimental import pipeline_def` (section 11.4). |
| `StopIteration` from a PyTorch DataLoader during your own benchmark | You asked for more batches than the dataset has (warm-up + measured). Imagenette train has only ~36 batches of 256. |
| `RuntimeError: ... cannot be moved from GPU to CPU` | A CPU operator is getting GPU data. Move all CPU operations **before** the first GPU or `mixed` operator. |
| Out of GPU memory after switching to DALI | DALI keeps its own memory pools. Lower `prefetch_queue_depth` or `batch_size`, or set `hw_decoder_load` / the decoder's memory padding. |
| Epoch is shorter or longer than expected | Pass `reader_name="Reader"` to the iterator, and use `pad_last_batch=True` when sharding. |
| All GPUs see the same images | You forgot `shard_id` / `num_shards`, or gave every rank the same `shard_id`. |
| Validation accuracy is slightly wrong | Use `LastBatchPolicy.PARTIAL` (not `FILL`) for validation. |
| Labels have shape `[B, 1]` | This is expected. Call `.squeeze(-1).long()`. |
| Training is still slow | Check whether the bottleneck is now **disk I/O** (`iostat`). Use local NVMe, `fn.readers.webdataset` shards, or GPUDirect Storage. |

## 16. When **not** to use DALI

- Your GPU is already at ~100% utilization. Then the input pipeline is not the bottleneck.
- The model itself is the bottleneck, for example very large models on small images.
- You depend on many custom PIL/NumPy augmentations that have no DALI equivalent. You can still use `fn.python_function`, but it is slower.
- You are not on Linux with an NVIDIA GPU.

## 17. Summary

1. Write the pipeline with `@pipeline_def` using `fn.*` operators.
2. Decode with `device="mixed"` so JPEG decoding runs on the GPU.
3. Prefer fused operators such as `image_random_crop` and `crop_mirror_normalize`.
4. Use `DALIGenericIterator` to get GPU tensors in PyTorch without copies.
5. Shard with `shard_id` / `num_shards` for multi-GPU training.
6. Use `external_source` for data that isn't stored as files.
7. **Benchmark** on your own hardware before and after. In my test, DALI loaded images 7–19× faster and video 1.3–1.9× faster, but sped up ResNet-50 training by only 8%, because the model was already the bottleneck. DALI helps most when the input pipeline is what is holding the GPU back.

### References
- Documentation: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
- GitHub and examples: https://github.com/NVIDIA/DALI
- Operator list: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html
