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
