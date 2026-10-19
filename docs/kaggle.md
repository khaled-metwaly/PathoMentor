# Running on Kaggle

The paper's runs were trained on Kaggle GPU sessions. The package runs there unchanged.

1. Create a notebook with **GPU** and **Internet** enabled.
2. Attach the tiled dataset (output of `teacher/make_patches.py`) and, for the mentor arm, the teacher cache
   (output of `teacher/build_teacher_cache.py`) as inputs.
3. In a code cell:

```python
!pip install -q git+https://github.com/khaled-metwaly/PathoMentor.git
!python -m pathomentor.train --arm mentor --fov 448 --seed 0 \
    --data-root /kaggle/input/<your-tiles-dataset>/bcss_patches_448 \
    --teacher-cache /kaggle/input/<your-teacher-output>/uni2h_native_softmaps_32x32x5.npy \
    --out-dir /kaggle/working/runs --max-hours 11
```

Use one notebook per (arm, seed) so every run gets its own session and time budget, then download the
`runs/` folders and run `python -m pathomentor.compare --runs <folder>` locally.

Notes
- The default `--max-hours 11` stops training (keeping the best weights) before Kaggle's 12-hour limit.
- Build the UNI2-h teacher in a separate GPU notebook; add your Hugging Face token as a Kaggle secret and
  pass it with `--hf-token` (or export `HF_TOKEN`). The cache is resumable across sessions.
- Training uses TensorFlow/Keras only; PyTorch is needed solely for the offline teacher.
