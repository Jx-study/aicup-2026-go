# AI CUP 2026 Go Rank and Style Prediction

> **English** | [繁體中文](README.md)

This repository holds our code for the AI CUP 2026 Go competition. It extends the official tutorial repository [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial).

## Tasks

| Task | Problem | Output |
|---|---|---|
| Task 1: player identification | Given several games by one player, find that player in the question's candidate pool | Top-5 candidates per question |
| Task 2: rank prediction | Given several games by one player, predict the player's rank | One rank per question (10 levels, D weakest, 6D strongest) |

- Player identification group A (`pub_`, `priv_`): 100 games per candidate, 5 to 20 games per question, all played as Black.
- Player identification group B (`pub2_`, `priv2_`): 10 games per candidate, 1 to 5 games per question, Black and White mixed.
- The official competition rules define the scoring.

## Repository layout

```
aicup-2026-go/
├── aicup/                     # Team code, run from the root as python -m aicup.<module>
├── unit_tests/                # Unit tests for aicup/
├── colab_*.ipynb              # Training, inference and KataGo analysis on Google Colab GPUs
├── network.py                 # Official: GoRankResNet, GoPlayerResNet
├── utils.py                   # Official: SGF parsing and feature extraction
├── rank-prediction.ipynb      # Official rank prediction tutorial, unchanged
├── player-identification.ipynb  # Official player identification tutorial, unchanged
├── justfile                   # Commands of the local steps
├── requirements.txt           # Python packages for the local steps
└── pytest.ini
```

- The official `network.py`, `utils.py` and the two tutorial notebooks are unchanged. All team methods live in `aicup/`.
- Data, features, checkpoints and outputs are not in the repository. They are created in folders under the root, such as `training-v1.1/`, `tests/`, `humansl-analysis/` and `player-trained-models/`.
- On Colab these folders live in Google Drive under `MyDrive/aicup-2026/`, with the same subfolder names as the repository root.

## Environment

- Local machine: data preparation, feature packing, score fusion and local evaluation, CPU only.
- Google Colab: model training, embedding inference and KataGo analysis, on GPU.

```bash
pip install -r requirements.txt
just test
```

- `requirements.txt` includes `rust-just`, which provides the `just` command. The full command of each step is in the `justfile` at the repository root: `just --list` lists the steps, and `just -n <step>` prints the commands without running them.
- On Windows the `justfile` runs its commands in PowerShell; on other systems it uses `sh`.
- `unit_tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK`, so the tests do not abort when PyTorch and another package both load OpenMP.

## Getting the data

| Data | Source | Location |
|---|---|---|
| Training set v1.1 | Releases of the official dataset repository [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset). Download `aicup2026_go_training_v1.1.tar.xz` and check the SHA-256 given in that README | Run `tar xf aicup2026_go_training_v1.1.tar.xz` inside `training-v1.1/` to get `training-v1.1/training/train_*.csv` |
| Test questions and candidate pools | The AI CUP competition platform, after registering for the competition | `tests/` |

| Task | Files in `tests/` (one public and one private each) |
|---|---|
| Rank prediction | `rank_prediction_test_{public,private}.csv` |
| Player identification group A | `player_identification_test_{public,private}.csv`, `player_identification_candidates_{public,private}.csv` |
| Player identification group B | `player_identification_test2_{public,private}.csv`, `player_identification_candidates2_{public,private}.csv` |

- Training set v1.1 adds the handicap tag `HD[n]` to `sgf_content`. All team code reads v1.1.

## Player split lists

```bash
just splits
```

- Splits are by player. A player is identified by `(rank, player_id)`. The random seed is fixed, so a rerun gives the same lists.
- `just splits` runs `python -m aicup.val_players --val-size 180000 --out-dir humansl-analysis` and skips it when the lists already exist.
- Every list has the columns `player_id` and `rank`.

| File | Content | Use |
|---|---|---|
| `val-players.csv` | Players with at least 20 games in the val part split by `--base-size` (default 100,000 games) | val648 evaluation set |
| `val-players-v180000.csv` | Players with at least 20 games in the val part split by `--val-size` | Union of the lists above and below |
| `val-players-new-v180000.csv` | The previous list minus `val-players.csv` | Source of the holdout set and newdev |
| `holdout-v180000.csv` | Holdout set drawn from the new players in proportion to each rank | Final veto test |
| `val-players-newdev-v180000.csv` | New players minus the holdout set | Development set: checkpoint and fusion weight selection |
| `val-players-dev-v180000.csv` | `val-players.csv` plus newdev | Development set for Task 2 |
| `train-pool-v180000.csv` | Players with at least 20 games in the train part split by `--val-size`, a fixed number per rank | Fitting data for the Human SL rank model |
| `train-p1-v180000.csv`, `train-p2-v180000.csv` | Players with at least 20 games in the whole training set who are not in dev, the holdout set, the train pool or `val-players.csv`; per rank, positions 1–100 and 101–200 after sorting by `player_id` | 2d pool of the Task 2 Transformer |
| `train-p3-v180000.csv` | Players with at least 20 games in the whole training set who are not in val, the holdout set, `train-p1` or `train-p2`; 100 per rank drawn with `random.Random(44)` | Same as above |
| `train-p4.csv` | Players in none of the lists above and not among players with at least 5 games in the base split's val, with at least 10 games of one color among games with HD < 2 | Same as above |

- The training set is the train part split by `--val-size`, so no val player in the table appears in the training data.
- `train-p1` to `train-p4` are drawn from the whole training set without the split; they are used only for Task 2.

## Reproducing the final Task 1 submission

| Track | How to run | Output |
|---|---|---|
| Opening fingerprint, local patterns | Computed from the game records inside `fuse_player` | None |
| Group A rank compatibility | `rank_cnn` in `colab_train.ipynb`, then `infer_player_rank` on the local machine | `player-trained-models/rank-cache/` |
| Group B rank compatibility | `trainpool_hist`, `val648_hist`, `newdev_hist` and `pub2` in `colab_humansl.ipynb`, then `infer_player_hsl` on the local machine | `player-trained-models/hsl-cache/` |
| Model B | `rank_cnn_v180000` in `colab_train.ipynb`, then `model_b`, `infer_a` and `infer_b` in `colab_player_emb.ipynb` | `player-trained-models/emb-model-b/emb-cache/` |
| Sequence models (7) | `colab_seq_emb.ipynb`, with `STAGE` set to `smoke`, `train` and `infer` in turn for each `ARM` | `player-trained-models/seq-<ARM>/emb-cache/` |

- The final score is a weighted sum of the similarities of the tracks above. The weights are in the `task1-submit` recipe of the `justfile`.

### 1. Local data preparation

```bash
just task1-prep
```

- `hsl_games` draws the games for the Human SL analysis from the player lists with seed 42 and writes `humansl-analysis/hist6d-<run>/games.csv`.
- `prep_player_hsl` writes the group B candidate and question games to `player-trained-models/hsl-cache/player_hsl_pub2_manifest.csv`.
- Upload `training-v1.1/`, `tests/`, `humansl-analysis/`, `rank-features/`, `rank-features-v180000/`, `move-seq-state-v180000/`, `move-seq-patch-v180000/` and `player-trained-models/hsl-cache/` to `MyDrive/aicup-2026/` in Google Drive, with the same paths as the repository root.

### 2. Colab

| Step | Notebook | Setting | Depends on |
|---|---|---|---|
| 1 | `colab_train.ipynb` | `MODE = 'rank_cnn'` | `rank-features/` |
| 2 | `colab_train.ipynb` | `MODE = 'rank_cnn_v180000'` | `rank-features-v180000/` |
| 3 | `colab_humansl.ipynb` | `MODE` set to `trainpool_hist`, `val648_hist`, `newdev_hist` and `pub2` in turn | `hist6d-<run>/games.csv`, `player_hsl_pub2_manifest.csv` |
| 4 | `colab_player_emb.ipynb` | `MODE` set to `model_b`, `infer_a` and `infer_b` in turn | Checkpoint from step 2 |
| 5 | `colab_seq_emb.ipynb` | For each of the 7 `ARM` values, `STAGE = 'smoke'`, `'train'` and `'infer'` in turn | `move-seq-state-v180000/`, `move-seq-patch-v180000/` |

- The "需要的檔案" (required files) section at the top of each notebook lists the Drive paths it reads and the commands that create them. The `## 準備資料` (prepare data) cell lists any missing files and stops.
- The notebooks clone the code from the public repository. Set `COMMIT` to a commit hash to pin a version.
- The notebooks download the KataGo binary and network files from the official sites, so they do not need to be on Drive: `b18c384nbt-humanv0` comes from the KataGo GitHub Releases, and the main network `kata1-b28c512nbt-s13255194368-d5935380940` used by `pub2` comes from the [KataGo networks page](https://katagotraining.org/networks/).
- After an interruption, rerun the same setting. The run resumes from `last.pt` or from the finished part of the jsonl on Drive.

### 3. Local inference and fusion

```bash
just task1-submit
```

- Before running, download `rank-trained-models/rank-cnn/`, `humansl-analysis/hist6d-*/` and `player-trained-models/` from Drive to the same paths under the repository root.
- `infer_player_rank` runs the group A CNN on the candidate and question games and writes rank distributions to `player-trained-models/rank-cache/`. It is slow on CPU only; a machine with a GPU is faster.
- `infer_player_hsl` fits the rank logistic regression on the target-side analysis of trainpool, val648 and newdev, then estimates the ranks of group B candidates and questions. It reads `humansl-analysis/hist6d-<run>/<run>-p3-hist.jsonl` and `player-trained-models/hsl-cache/pub2-p3-hist.jsonl`.
- A KataGo rerun has small noise and GPU training is not bit-for-bit reproducible, so a rerun submission can differ from the team's upload on a few questions.

## Reproducing the final Task 2 submission

The final Task 2 submission is (A + B + TF) / 3: A and B are rank logistic regressions on Human SL features, and TF is a Transformer on per-move features. Their 10-level probabilities are averaged and the level is chosen by expected score.

### 1. Drawing games locally

```bash
just task2-games
```

- Runs with `--2d` are written to `humansl-analysis/hist2d-<run>/games.csv`. `train_p4` takes 10 games of each player's majority color among games with HD < 2, and `train_p4b` draws 10 more unused games of the same color from the same players.
- The 2d test questions need no games.csv; the analysis reads `tests/` directly.
- Upload `training-v1.1/`, `humansl-analysis/` and `tests/` to `MyDrive/aicup-2026/` on Google Drive.

### 2. Colab: Human SL analysis

| Order | Notebook | `MODE` | Output |
|---|---|---|---|
| 1 | `colab_humansl.ipynb` | `val648` | `humansl-analysis/val648-p3.jsonl`, the reference for the 2d control check |
| 2 | `colab_humansl.ipynb` | `trainpool_hist`, `newdev_hist`, `val648_hist`, `test_hist` and their `_opp` | `humansl-analysis/hist6d-<run>/` |
| 3 | `colab_humansl.ipynb` | `train_p1_2d`, `train_p2_2d`, `train_p3_2d`, `train_p4_2d`, `train_p4b_2d`, `val648_2d`, `test_2d` and their `_opp` | `humansl-analysis/hist2d-<run>/` |

- Before the analysis, 2d MODEs run 40 control games and compare them with `val648-p3.jsonl`; they stop when the median absolute difference exceeds 0.03.
- The control check reads only the first 40 games of `val648-p3.jsonl`, so the `val648` run can be stopped once its first batch is backed up to Drive.
- 2d MODEs use the main network to compute per-move loss; the notebook downloads `kata1-b28c512nbt-s13255194368-d5935380940`.

### 3. Packing locally

```bash
just task2-pack
```

- Download the jsonl files under `humansl-analysis/hist6d-*/` and `hist2d-*/` from Drive first, then upload the packed `packed-moves/` and `packed-moves-2d/`.
- `train_p4all` concatenates `train_p4` and `train_p4b` in that order.

### 4. Locally: models A and B

```bash
just task2-ab
```

- A uses the 6d profiles: the development set is val648 plus newdev, fitted together with the train pool. B uses the 2d profiles plus per-move loss features of both sides: the development set is val648, fitted together with train_p1 and train_p2.
- Both use only the first-stage logistic regression (`--s1-only`), fitted on same-color questions with HD < 2 (`--same-color`), and need no CNN.
- CPU only; each model takes minutes to tens of minutes.

### 5. Colab: training TF

| Notebook | `MODE` | Output |
|---|---|---|
| `colab_tf.ipynb` | `smoke` first, then `j30_p4all` | `humansl-analysis/tf-final-j30-p4all/out/final-ts{0,1,2}-test.npy` |

### 6. Fusion locally

```bash
just task2-submit
```

- `--tf-weight` defaults to 1/3, which is the plain average of the three.

## Experimental notebooks and modules

| Notebook | Content |
|---|---|
| `colab_permove.ipynb` | Fine-tunes a per-move rank model on KataGo humanv0 (`train_permove`, `infer_permove`) |
| `colab_setmodel.ipynb` | Extracts humanv0 trunk vectors for the set model probe (`setmodel_trunk`) |
| `colab_strength_se.ipynb` | Inference with the Academia Sinica strength estimator (`strength_se`) |
| `colab_trunk.ipynb` | Discriminative fine-tuning of the humanv0 trunk (`trunk_cls`); questions from `export_questions`, games from `trunk_games` |

- The 4 notebooks above are Task 2 experiments and are not part of the final pipeline of either task.
- `6d` and `2d` in `colab_tf.ipynb` are earlier Task 2 TF versions and are not part of the final pipeline.
- `aicup.export_questions` generates the fixed val648 evaluation questions; the `pool` and `cv` modes of `tf_run.py` need `questions-s{47..51}.npz` in the `--data` directory.
- The following MODEs are not part of the final Task 1 pipeline either:
  - `colab_humansl.ipynb`: `*_hist_opp`, `pub2_opp`, `val648`, `val648_opp`, `valnew`, `valnew_opp`, `trainpool`, `trainpool_opp`, `test`, `test_opp`, `test_hist*`, `*_2d*`, `pilot`, `visits`, `main`.
  - `colab_train.ipynb`: `e2e_smoke`, `e2e_full`.
  - `colab_player_emb.ipynb`: `full`, `stage2`.
  - `colab_seq_emb.ipynb`: `ARM = 'patch_conv_wide'`.

| Experiment in report section 陸 (analysis and conclusions) | Code |
|---|---|
| Larger sequence model (PV6.2) | `ARM = 'patch_conv_wide'` in `colab_seq_emb.ipynb` |
| Local simulated questions (PV2) | `aicup/eval_player_sim.py` |

## Acknowledgements and license

- Serkan Kavak of NDHU AI Lab made the official tutorial repository. `network.py`, `utils.py` and the two tutorial notebooks come from it.
- The official tutorial repository lets participants use, modify and distribute its code freely.
- The training set is for 2026 AI CUP competition use only and must not be used to identify real users.
