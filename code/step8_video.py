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
