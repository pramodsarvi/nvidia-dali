# step6_benchmark.py
import time, torch, torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from train_dali import create_pipeline   # reuse the pipeline from Step 4

DATA, BS, WORKERS, N_BATCHES = "imagenette2-320/train", 256, 8, 30  # Imagenette train has 9,469 imgs (~36 batches); keep warm-up + N_BATCHES below that

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
