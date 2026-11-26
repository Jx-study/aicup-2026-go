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

## Acknowledgements and license

- Serkan Kavak of NDHU AI Lab made the official tutorial repository. `network.py`, `utils.py` and the two tutorial notebooks come from it.
- The official tutorial repository lets participants use, modify and distribute its code freely.
- The training set is for 2026 AI CUP competition use only and must not be used to identify real users.
