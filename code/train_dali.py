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
