# step7_advanced.py - conditionals, RandAugment, debug mode (see blog section 11)
import sys
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.auto_aug import rand_augment

MEAN = [123.7, 116.3, 103.5]; STD = [58.4, 57.1, 57.4]

@pipeline_def(enable_conditionals=True)
def conditional_aug(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed")
    if fn.random.coin_flip(probability=0.3):
        images = fn.rotate(images, angle=fn.random.uniform(range=[-30, 30]), fill_value=0)
    else:
        images = fn.hsv(images, hue=fn.random.uniform(range=[-20, 20]))
    images = fn.resize(images, resize_x=224, resize_y=224)
    return images, labels.gpu()

@pipeline_def(enable_conditionals=True)
def randaug_pipeline(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = rand_augment.rand_augment(images, n=2, m=9)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout="CHW",
                                      mean=MEAN, std=STD)
    return images, labels.gpu()

@pipeline_def(debug=True)
def debug_pipeline(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir)
    images = fn.decoders.image(jpegs, device="cpu")
    print("debug shapes:", images.shape()[:4])
    return images, labels

if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "imagenette2-320/train"
    for factory in (conditional_aug, randaug_pipeline, debug_pipeline):
        p = factory(data_dir=data, batch_size=16, num_threads=4, device_id=0, seed=0)
        p.build()
        out = p.run()
        print(factory.__name__, "->", len(out[0]), "samples")
