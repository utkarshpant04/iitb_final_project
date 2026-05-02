# GNR638 Project (22b0914, 22b0932, 22b0989)  

Solves PNG images of deep learning multiple-choice questions using
**Qwen2.5-VL-7B-Instruct**, a vision-language model with strong OCR and
reasoning capabilities.
---

## Setup

```bash
bash setup.bash          # clones repo, creates env, downloads weights
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

---

## Expected directory layout

```
<working_dir>/
├── inference.py
├── download_model.py
├── requirements.txt
├── saved_qwen25vl_7b/      ← downloaded by download_model.py
└── submission.csv          ← created by inference.py
```

```
<test_dir>/
├── images/
│   ├── image_1.png
│   └── ...
└── test.csv
```

Output `submission.csv` is written to the working directory.
## Honor Code

- Claude Code used for debugging purposes.
