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
