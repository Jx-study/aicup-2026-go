# AI CUP 2026 圍棋棋力與風格預測競賽：TEAM_12242

> [English](README.en.md) | **繁體中文**

本 repository 是 TEAM_12242 參加 AI CUP 2026 圍棋競賽的程式碼，從官方教學 repository [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial) 延伸而來。

## 競賽題目

| 競賽 | 題目 | 輸出 |
|---|---|---|
| 競賽一：玩家辨識 | 給一位玩家的若干局棋譜，從該題所屬的候選庫找出這位玩家 | 每題 Top-5 候選 |
| 競賽二：棋力預測 | 給一位玩家的若干局棋譜，預測他的棋力等級 | 每題 1 個等級（10 級，D 最弱、6D 最強） |

- 玩家辨識 A 組（`pub_`、`priv_`）：每位候選 100 局，每題 5～20 局，全部執黑。
- 玩家辨識 B 組（`pub2_`、`priv2_`）：每位候選 10 局，每題 1～5 局，黑白混合。
- 計分方式以官方說明為準。

## 檔案結構

```
aicup-2026-go/
├── aicup/                     # 本隊的程式，從根目錄以 python -m aicup.<module> 執行
├── unit_tests/                # aicup/ 的單元測試
├── colab_*.ipynb              # 在 Google Colab GPU 上執行訓練、推論與 KataGo 分析
├── network.py                 # 官方：GoRankResNet、GoPlayerResNet
├── utils.py                   # 官方：SGF 解析與特徵抽取
├── rank-prediction.ipynb      # 官方棋力預測教學 notebook，未修改
├── player-identification.ipynb  # 官方玩家辨識教學 notebook，未修改
└── pytest.ini
```

- 官方的 `network.py`、`utils.py` 與兩份教學 notebook 保持原樣，本隊的做法都放在 `aicup/`。
- 資料、特徵、checkpoint 與輸出檔不放在 repository，執行時產生在根目錄下的資料夾，例如 `training-v1.1/`、`tests/`、`humansl-analysis/`、`player-trained-models/`。
- 在 Colab 上執行時，這些資料夾放在 Google Drive 的 `MyDrive/aicup-2026/`，子目錄名稱與 repository 根目錄相同。

## 執行環境

- 本機：資料準備、特徵打包、融合推論與本地評估，只用 CPU。
- Google Colab：模型訓練、嵌入推論與 KataGo 分析，使用 GPU。

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib sgfmill pytest
pytest
```

- `unit_tests/conftest.py` 設定 `KMP_DUPLICATE_LIB_OK`，避免 PyTorch 與其他套件重複載入 OpenMP 時中止。

## 取得資料

| 資料 | 取得方式 | 放置位置 |
|---|---|---|
| 訓練集 v1.1 | 官方資料集 repository [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset) 的 Releases，下載 `aicup2026_go_training_v1.1.tar.xz`，核對該 README 上的 SHA-256 | 在 `training-v1.1/` 內執行 `tar xf aicup2026_go_training_v1.1.tar.xz`，得到 `training-v1.1/training/train_*.csv` |
| 測試題與候選庫 | AI CUP 競賽平台，需報名參賽 | `tests/` |

| 題目 | `tests/` 內的檔案（public 與 private 各一份） |
|---|---|
| 棋力預測 | `rank_prediction_test_{public,private}.csv` |
| 玩家辨識 A 組 | `player_identification_test_{public,private}.csv`、`player_identification_candidates_{public,private}.csv` |
| 玩家辨識 B 組 | `player_identification_test2_{public,private}.csv`、`player_identification_candidates2_{public,private}.csv` |

- 訓練集 v1.1 在 `sgf_content` 加上讓子資訊 `HD[n]`，本隊的程式一律讀 v1.1。

## 玩家切分清單

```bash
python -m aicup.val_players --val-size 180000 --out-dir humansl-analysis
```

- 切分以玩家為單位，用 `(rank, player_id)` 識別玩家，亂數種子固定，重跑得到相同的清單。
- 輸出目錄已有同名檔案時不覆寫，直接停止。
- 每份清單的欄位都是 `player_id`、`rank`。

| 檔案 | 內容 | 用途 |
|---|---|---|
| `val-players.csv` | `--base-size`（預設 100,000 局）切出的 val 中，局數至少 20 的玩家 | val648 評估集 |
| `val-players-v180000.csv` | `--val-size` 切出的 val 中，局數至少 20 的玩家 | 上下兩份的聯集 |
| `val-players-new-v180000.csv` | 上一份扣掉 `val-players.csv` | 封存組與 newdev 的來源 |
| `holdout-v180000.csv` | 從新增玩家依等級比例抽出的封存組 | 最後的否決檢定 |
| `val-players-newdev-v180000.csv` | 新增玩家扣掉封存組 | 開發集：選 checkpoint 與融合權重 |
| `val-players-dev-v180000.csv` | `val-players.csv` 加上 newdev | 競賽二的開發集 |
| `train-pool-v180000.csv` | `--val-size` 切出的 train 中，局數至少 20 的玩家，每級固定人數 | Human SL 段位模型的擬合資料 |

- 訓練集由 `--val-size` 切出的 train 部分構成，所以上表的 val 玩家都不會出現在訓練資料中。

## 重現競賽一最終版本

| 軌 | 執行 | 輸出 |
|---|---|---|
| 開局指紋、局部棋形 | `fuse_player` 內直接從棋譜計算 | 無 |
| A 組段位相容度 | `colab_train.ipynb` 的 `rank_cnn`，再在本機執行 `infer_player_rank` | `player-trained-models/rank-cache/` |
| B 組段位相容度 | `colab_humansl.ipynb` 的 `trainpool_hist`、`val648_hist`、`newdev_hist`、`pub2`，再在本機執行 `infer_player_hsl` | `player-trained-models/hsl-cache/` |
| Model B | `colab_train.ipynb` 的 `rank_cnn_v180000`，再執行 `colab_player_emb.ipynb` 的 `model_b`、`infer_a`、`infer_b` | `player-trained-models/emb-model-b/emb-cache/` |
| 序列模型（7 個） | `colab_seq_emb.ipynb`，每個 `ARM` 依序執行 `STAGE` 的 `smoke`、`train`、`infer` | `player-trained-models/seq-<ARM>/emb-cache/` |

- 最終版本的分數是上表各軌相似度的加權和，權重見第 3 步的 `fuse_player` 指令。

### 1. 本機資料準備

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

- `hsl_games` 依玩家清單與 seed 42 抽出 Human SL 分析的局，寫成 `humansl-analysis/hist6d-<run>/games.csv`。
- `prep_player_hsl` 把 B 組的候選局與題目局寫成 `player-trained-models/hsl-cache/player_hsl_pub2_manifest.csv`。
- 把 `training-v1.1/`、`tests/`、`humansl-analysis/`、`rank-features/`、`rank-features-v180000/`、`move-seq-state-v180000/`、`move-seq-patch-v180000/`、`player-trained-models/hsl-cache/` 上傳到 Google Drive 的 `MyDrive/aicup-2026/`，路徑和 repository 根目錄相同。

### 2. Colab

| 順序 | notebook | 設定 | 依賴 |
|---|---|---|---|
| 1 | `colab_train.ipynb` | `MODE = 'rank_cnn'` | `rank-features/` |
| 2 | `colab_train.ipynb` | `MODE = 'rank_cnn_v180000'` | `rank-features-v180000/` |
| 3 | `colab_humansl.ipynb` | `MODE` 依序為 `trainpool_hist`、`val648_hist`、`newdev_hist`、`pub2` | `hist6d-<run>/games.csv`、`player_hsl_pub2_manifest.csv` |
| 4 | `colab_player_emb.ipynb` | `MODE` 依序為 `model_b`、`infer_a`、`infer_b` | 第 2 步的 checkpoint |
| 5 | `colab_seq_emb.ipynb` | 7 個 `ARM` 各自依序執行 `STAGE = 'smoke'`、`'train'`、`'infer'` | `move-seq-state-v180000/`、`move-seq-patch-v180000/` |

- 每份 notebook 開頭的「需要的檔案」列出它讀的 Drive 路徑與產生指令，`## 準備資料` 格缺檔時會列出缺少的檔案後停止。
- notebook 從公開的 repository clone 程式，`COMMIT` 可以填 commit hash 固定版本。
- KataGo 執行檔與網路檔由 notebook 從官方網址下載，不用放 Drive：`b18c384nbt-humanv0` 來自 KataGo 的 GitHub Releases，`pub2` 用的主網路 `kata1-b28c512nbt-s13255194368-d5935380940` 來自 [KataGo 網路下載頁](https://katagotraining.org/networks/)。
- 中斷時重跑同一個設定，會從 Drive 上的 `last.pt` 或已完成的 jsonl 接續。

### 3. 本機推論與融合

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

- 執行前從 Drive 下載 `rank-trained-models/rank-cnn/`、`humansl-analysis/hist6d-*/`、`player-trained-models/` 到 repository 根目錄的同名路徑。
- `infer_player_rank` 用 A 組的 CNN 推論候選與題目的段位分布，寫到 `player-trained-models/rank-cache/`；只用 CPU 時較慢，可改在有 GPU 的環境執行。
- `infer_player_hsl` 用 trainpool、val648、newdev 的目標側分析擬合段位邏輯回歸，再估計 B 組候選與題目的段位，讀 `humansl-analysis/hist6d-<run>/<run>-p3-hist.jsonl` 與 `player-trained-models/hsl-cache/pub2-p3-hist.jsonl`。
- KataGo 重跑有小雜訊，GPU 訓練也不是逐位元可重現，所以重跑的提交檔和本隊上傳的版本可能有少數題目不同。

## 實驗性 notebook 與模組

| notebook | 內容 |
|---|---|
| `colab_permove.ipynb` | 微調 KataGo humanv0 的逐手段位模型（`train_permove`、`infer_permove`） |
| `colab_setmodel.ipynb` | 抽取 humanv0 主幹向量，供 set 模型探針使用（`setmodel_trunk`） |
| `colab_strength_se.ipynb` | 中研院 strength estimator 推論（`strength_se`） |

- 上表 3 份是競賽二的實驗，不在競賽一最終版本流程內。
- 以下 MODE 同樣不在競賽一最終版本流程內：
  - `colab_humansl.ipynb`：`*_hist_opp`、`pub2_opp`、`val648`、`val648_opp`、`valnew`、`valnew_opp`、`trainpool`、`trainpool_opp`、`test`、`test_opp`、`pilot`、`visits`、`main`。
  - `colab_train.ipynb`：`e2e_smoke`、`e2e_full`。
  - `colab_player_emb.ipynb`：`full`、`stage2`。
  - `colab_seq_emb.ipynb`：`ARM = 'patch_conv_wide'`。

| 報告「陸、分析與結論」的實驗 | 程式 |
|---|---|
| 加大序列模型容量（PV6.2） | `colab_seq_emb.ipynb` 的 `ARM = 'patch_conv_wide'` |
| 本地模擬題（PV2） | `aicup/eval_player_sim.py` |

## 致謝與授權

- 官方教學 repository 由 NDHU AI Lab 的 Serkan Kavak 製作，`network.py`、`utils.py` 與兩份教學 notebook 來自該 repository。
- 官方教學 repository 允許參賽者自由使用、修改與散布其程式碼。
- 訓練集僅供 2026 AI CUP 競賽使用，不得用於識別真實使用者。
