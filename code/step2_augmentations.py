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
