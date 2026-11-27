# AI CUP 2026 Go Rank and Style Prediction: TEAM_12242

> **English** | [繁體中文](README.md)

This repository holds the code of TEAM_12242 for the AI CUP 2026 Go competition. It extends the official tutorial repository [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial).

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
└── pytest.ini
```

- The official `network.py`, `utils.py` and the two tutorial notebooks are unchanged. All team methods live in `aicup/`.
- Data, features, checkpoints and outputs are not in the repository. They are created in folders under the root, such as `training-v1.1/`, `tests/`, `humansl-analysis/` and `player-trained-models/`.
- On Colab these folders live in Google Drive under `MyDrive/aicup-2026/`, with the same subfolder names as the repository root.

## Environment

- Local machine: data preparation, feature packing, score fusion and local evaluation, CPU only.
- Google Colab: model training, embedding inference and KataGo analysis, on GPU.

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib sgfmill pytest
pytest
```

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
python -m aicup.val_players --val-size 180000 --out-dir humansl-analysis
```

- Splits are by player. A player is identified by `(rank, player_id)`. The random seed is fixed, so a rerun gives the same lists.
- The command stops without overwriting if the output folder already has a file with the same name.
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

- The training set is the train part split by `--val-size`, so no val player in the table appears in the training data.

## Reproducing the final Task 1 submission

| Track | How to run | Output |
|---|---|---|
| Opening fingerprint, local patterns | Computed from the game records inside `fuse_player` | None |
| Group A rank compatibility | `rank_cnn` in `colab_train.ipynb`, then `infer_player_rank` on the local machine | `player-trained-models/rank-cache/` |
| Group B rank compatibility | `trainpool_hist`, `val648_hist`, `newdev_hist` and `pub2` in `colab_humansl.ipynb`, then `infer_player_hsl` on the local machine | `player-trained-models/hsl-cache/` |
| Model B | `rank_cnn_v180000` in `colab_train.ipynb`, then `model_b`, `infer_a` and `infer_b` in `colab_player_emb.ipynb` | `player-trained-models/emb-model-b/emb-cache/` |
| Sequence models (7) | `colab_seq_emb.ipynb`, with `STAGE` set to `smoke`, `train` and `infer` in turn for each `ARM` | `player-trained-models/seq-<ARM>/emb-cache/` |

- The final score is a weighted sum of the similarities of the tracks above. The weights are in the `fuse_player` command of step 3.

### 1. Local data preparation

```bash
python -m aicup.val_players --val-size 180000 --out-dir humansl-analysis
python -m aicup.extract_features --train-dir training-v1.1/training
python -m aicup.extract_features --train-dir training-v1.1/training --val-size 180000 --out rank-features-v180000
python -m aicup.color --train-dir training-v1.1/training --out rank-features-v180000/hd.npz
python -m aicup.hsl_games --run trainpool val648 newdev
python -m aicup.prep_player_hsl --tests-dir tests
python -m aicup.pack_move_seq --state-feats --out-dir move-seq-state-v180000
python -m aicup.pack_move_seq --state-feats --extra-split hold=humansl-analysis/holdout-v180000.csv --splits hold --pat3-vocab move-seq-state-v180000/pat3_vocab.npy --out-dir move-seq-state-v180000
python -m aicup.pack_move_patch --seq-dir move-seq-state-v180000 --out-dir move-seq-patch-v180000
python -m aicup.pack_move_patch --seq-dir move-seq-state-v180000 --out-dir move-seq-patch-v180000 --splits hold
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
python -m aicup.infer_player_rank --ckpt rank-trained-models/rank-cnn/best_model.pth
python -m aicup.infer_player_hsl
python -m aicup.fuse_player --color-cond --pattern-beta 0.7 \
  --cand2-cache player-trained-models/hsl-cache/cand2_hsl_ranks.json \
  --query2-cache player-trained-models/hsl-cache/query2_hsl_ranks.json \
  --emb-gamma 1.0 --csls-k 5 --emb-top2-weight 0.5 --raw-open \
  --emb-cache-dir player-trained-models/emb-model-b/emb-cache \
  --seq-cache-dir player-trained-models/seq-state/emb-cache \
                  player-trained-models/seq-state_conf/emb-cache \
                  player-trained-models/seq-patch/emb-cache \
                  player-trained-models/seq-patch_conv/emb-cache \
                  player-trained-models/seq-patch_crop/emb-cache \
                  player-trained-models/seq-patch_d4/emb-cache \
                  player-trained-models/seq-patch_d4_val648/emb-cache \
  --seq-groups all --seq-weights 0.5 0.5 0.1 0.1 0.1 0.1 0.1 \
  --pub2-alpha 0.5 --pub2-emb-gamma 0.6 --pub2-open-weight 0.0 \
  --pub2-seq-weights 0.3 0.3 0.4 0.4 0.4 0.4 0.4 \
  --out-csv submission_player.csv
```

- Before running, download `rank-trained-models/rank-cnn/`, `humansl-analysis/hist6d-*/` and `player-trained-models/` from Drive to the same paths under the repository root.
- `infer_player_rank` runs the group A CNN on the candidate and question games and writes rank distributions to `player-trained-models/rank-cache/`. It is slow on CPU only; a machine with a GPU is faster.
- `infer_player_hsl` fits the rank logistic regression on the target-side analysis of trainpool, val648 and newdev, then estimates the ranks of group B candidates and questions. It reads `humansl-analysis/hist6d-<run>/<run>-p3-hist.jsonl` and `player-trained-models/hsl-cache/pub2-p3-hist.jsonl`.
- A KataGo rerun has small noise and GPU training is not bit-for-bit reproducible, so a rerun submission can differ from the team's upload on a few questions.

## Experimental notebooks and modules

| Notebook | Content |
|---|---|
| `colab_permove.ipynb` | Fine-tunes a per-move rank model on KataGo humanv0 (`train_permove`, `infer_permove`) |
| `colab_setmodel.ipynb` | Extracts humanv0 trunk vectors for the set model probe (`setmodel_trunk`) |
| `colab_strength_se.ipynb` | Inference with the Academia Sinica strength estimator (`strength_se`) |

- The 3 notebooks above are Task 2 experiments and are not part of the final Task 1 pipeline.
- The following MODEs are not part of the final Task 1 pipeline either:
  - `colab_humansl.ipynb`: `*_hist_opp`, `pub2_opp`, `val648`, `val648_opp`, `valnew`, `valnew_opp`, `trainpool`, `trainpool_opp`, `test`, `test_opp`, `pilot`, `visits`, `main`.
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
