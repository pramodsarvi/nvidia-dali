# NVIDIA DALI: End-to-End Tutorial

Companion code for the blog post [`blog.md`](blog.md): fast GPU data loading with NVIDIA DALI, covering images, video, PyTorch training, and benchmarking.

> Requires Linux + NVIDIA GPU. `pip install -r code/requirements.txt`

| File | What it shows |
|---|---|
| `code/step1_first_pipeline.py` | First pipeline: read → GPU decode → resize |
| `code/step2_augmentations.py` | Train/val augmentation pipelines |
| `code/train_dali.py` | End-to-end ResNet-50 training (single/multi-GPU) |
| `code/step5_external_source.py` | Custom data sources (CSV) |
| `code/step6_benchmark.py` | Quick DataLoader vs DALI comparison |
| `code/step7_advanced.py` | Conditionals, RandAugment, debug mode |
| `code/step8_video.py` | Video ingestion and action recognition training |
| `code/benchmark.py` | Full image/video benchmark suite |
| `code/plot_results.py` | Plot benchmark results |
