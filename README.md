# AI CUP 2026 圍棋棋力與風格預測競賽

> [English](README.en.md) | **繁體中文**

本 repository 是 AI CUP 2026 圍棋競賽的參賽程式碼，從官方教學 repository [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial) 延伸而來。

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
├── rank-prediction.ipynb      # 官方棋力預測教學 notebook
├── player-identification.ipynb  # 官方玩家辨識教學 notebook
├── justfile                   # 本機各步驟的指令
├── requirements.txt           # 本機需要的套件
└── pytest.ini
```

- 官方的 `network.py`、`utils.py` 與兩份教學 notebook 保持原樣，本隊的做法都放在 `aicup/`。
- 資料、特徵、checkpoint 與輸出檔不放在 repository，執行時產生在根目錄下的資料夾，例如 `training-v1.1/`、`tests/`、`humansl-analysis/`、`player-trained-models/`。
- 在 Colab 上執行時，這些資料夾放在 Google Drive 的 `MyDrive/aicup-2026/`，子目錄名稱與 repository 根目錄相同；各步驟寫明它之後要上傳或之前要下載的資料夾。

## 執行環境

- 本機：資料準備、特徵打包、融合推論與本地評估，只用 CPU。
- Google Colab：模型訓練、嵌入推論與 KataGo 分析，使用 GPU。

```bash
pip install -r requirements.txt
just test
```

- `requirements.txt` 含 `rust-just`，安裝後即可使用 `just` 指令；各步驟的完整指令寫在根目錄的 `justfile`，`just --list` 列出全部步驟，`just -n <步驟>` 只印出指令不執行。
- 在 Windows 上 `justfile` 以 PowerShell 執行，其他系統用 `sh`。

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
just splits
```

- 切分以玩家為單位，用 `(rank, player_id)` 識別玩家，亂數種子固定，重跑得到相同的清單；清單已存在時略過。
- 每份清單寫在 `humansl-analysis/`，欄位都是 `player_id`、`rank`。

| 檔案 | 內容 | 用途 |
|---|---|---|
| `val-players.csv` | `--base-size`（預設 100,000 局）切出的 val 中，局數至少 20 的玩家 | val648 評估集 |
| `val-players-v180000.csv` | `--val-size` 切出的 val 中，局數至少 20 的玩家 | 上下兩份的聯集 |
| `val-players-new-v180000.csv` | 上一份扣掉 `val-players.csv` | 封存組與 newdev 的來源 |
| `holdout-v180000.csv` | 從新增玩家依等級比例抽出的封存組 | 最後的否決檢定 |
| `val-players-newdev-v180000.csv` | 新增玩家扣掉封存組 | 開發集：選 checkpoint 與融合權重 |
| `val-players-dev-v180000.csv` | `val-players.csv` 加上 newdev | 競賽二的開發集 |
| `train-pool-v180000.csv` | `--val-size` 切出的 train 中，局數至少 20 的玩家，每級固定人數 | Human SL 段位模型的擬合資料 |
| `train-p1-v180000.csv`、`train-p2-v180000.csv` | 全部訓練局中局數至少 20、不在 dev、封存組、train 池、`val-players.csv` 的玩家，每級依 `player_id` 排序後的第 1～100、101～200 位 | 競賽二 Transformer 的 2d 池 |
| `train-p3-v180000.csv` | 全部訓練局中局數至少 20、不在 val、封存組、`train-p1`、`train-p2` 的玩家，每級用 `random.Random(44)` 抽 100 位 | 同上 |
| `train-p4.csv` | 不在以上清單、也不在基準切分 val 中局數至少 5 的玩家，HD < 2 的局中同一色至少 10 局 | 同上 |

- 訓練集由 `--val-size` 切出的 train 部分構成，所以上表的 val 玩家都不會出現在訓練資料中。
- `train-p1`～`train-p4` 從全部訓練局選，不經切分；它們只用於競賽二。

## 重現競賽一最終版本

| 軌 | 執行 | 輸出 |
|---|---|---|
| 開局指紋、局部棋形 | `fuse_player` 內直接從棋譜計算 | 無 |
| A 組段位相容度 | `colab_train.ipynb` 的 `rank_cnn`，再在本機執行 `infer_player_rank` | `player-trained-models/rank-cache/` |
| B 組段位相容度 | `colab_humansl.ipynb` 的 `trainpool_hist`、`val648_hist`、`newdev_hist`、`pub2`，再在本機執行 `infer_player_hsl` | `player-trained-models/hsl-cache/` |
| Model B | `colab_train.ipynb` 的 `rank_cnn_v180000`，再執行 `colab_player_emb.ipynb` 的 `model_b`、`infer_a`、`infer_b` | `player-trained-models/emb-model-b/emb-cache/` |
| 序列模型（7 個） | `colab_seq_emb.ipynb`，每個 `ARM` 依序執行 `STAGE` 的 `smoke`、`train`、`infer` | `player-trained-models/seq-<ARM>/emb-cache/` |

- 最終版本的分數是上表各軌相似度的加權和，權重見 `justfile` 的 `task1-submit`。

### 1. 本機資料準備

```bash
just task1-prep
```

- 完成後把 `training-v1.1/`、`tests/`、`humansl-analysis/`、`rank-features/`、`rank-features-v180000/`、`move-seq-state-v180000/`、`move-seq-patch-v180000/`、`player-trained-models/hsl-cache/` 上傳到 Drive 的 `MyDrive/aicup-2026/`，路徑和 repository 根目錄相同，例如 `rank-features/` 放在 `MyDrive/aicup-2026/rank-features/`。

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
just task1-submit
```

- 執行前從 `MyDrive/aicup-2026/` 下載 `rank-trained-models/rank-cnn/`、`humansl-analysis/hist6d-*/`、`player-trained-models/` 到 repository 根目錄的同名路徑。
- 其中的 CNN 段位推論（`infer_player_rank`）只用 CPU 時較慢，可改在有 GPU 的環境執行。
- KataGo 重跑有小雜訊，GPU 訓練也不是逐位元可重現，所以重跑的提交檔和本隊上傳的版本可能有少數題目不同。

## 重現競賽二最終版本

競賽二最終版本是 (A + B + TF) / 3：A、B 是 Human SL 特徵的段位邏輯回歸，TF 是逐手特徵的 Transformer，三者的 10 級機率平均後用期望得分選等級。

### 1. 本機抽局

```bash
just task2-games
```

- 2d 的局寫到 `humansl-analysis/hist2d-<run>/games.csv`；測試題分析時直接讀 `tests/`。
- 上傳 `training-v1.1/`、`humansl-analysis/`、`tests/` 到 `MyDrive/aicup-2026/` 的同名路徑。

### 2. Colab：Human SL 分析

| 順序 | notebook | `MODE` | 輸出 |
|---|---|---|---|
| 1 | `colab_humansl.ipynb` | `val648` | `humansl-analysis/val648-p3.jsonl`，2d MODE 對照組的參考 |
| 2 | `colab_humansl.ipynb` | `trainpool_hist`、`newdev_hist`、`val648_hist`、`test_hist` 與各自的 `_opp` | `humansl-analysis/hist6d-<run>/` |
| 3 | `colab_humansl.ipynb` | `train_p1_2d`、`train_p2_2d`、`train_p3_2d`、`train_p4_2d`、`train_p4b_2d`、`val648_2d`、`test_2d` 與各自的 `_opp` | `humansl-analysis/hist2d-<run>/` |

- 2d MODE 分析前先跑 40 局對照組，和 `val648-p3.jsonl` 比對；中位數絕對差超過 0.03 時停止。
- 對照組只讀 `val648-p3.jsonl` 的前 40 局，`val648` 第一批備份到 Drive 後就可以停止。

### 3. 本機打包

```bash
just task2-pack
```

- 先從 `MyDrive/aicup-2026/` 下載 `humansl-analysis/hist6d-*/`、`hist2d-*/` 的 jsonl，打包結果 `humansl-analysis/packed-moves/`、`packed-moves-2d/` 再上傳到同名路徑。

### 4. 本機：A、B 模型

```bash
just task2-ab
```

- A 用 6d profile：開發組是 val648 加 newdev，加 train 池擬合。B 用 2d profile 加兩側的每手損失特徵：開發組是 val648，加 train_p1、train_p2 擬合。
- 兩者都只用第一段邏輯回歸，擬合題是同色、HD < 2 的題，不需要 CNN；只用 CPU，每個模型數分鐘到數十分鐘。

### 5. Colab：訓練 TF

| notebook | `MODE` | 輸出 |
|---|---|---|
| `colab_tf.ipynb` | 先 `smoke`，再 `j30_p4all` | `humansl-analysis/tf-final-j30-p4all/out/final-ts{0,1,2}-test.npy` |

### 6. 本機融合

```bash
just task2-submit
```

- 執行前從 `MyDrive/aicup-2026/` 下載 `humansl-analysis/tf-final-j30-p4all/out/` 到同名路徑。

## 實驗性 notebook 與模組

| notebook | 內容 |
|---|---|
| `colab_permove.ipynb` | 微調 KataGo humanv0 的逐手段位模型（`train_permove`、`infer_permove`） |
| `colab_setmodel.ipynb` | 抽取 humanv0 主幹向量，供 set 模型探針使用（`setmodel_trunk`） |
| `colab_strength_se.ipynb` | 中研院 strength estimator 推論（`strength_se`） |
| `colab_trunk.ipynb` | humanv0 主幹判別式微調（`trunk_cls`），題目由 `export_questions`、棋譜由 `trunk_games` 產生 |

- 上表 4 份是競賽二的實驗，不在兩個競賽的最終版本流程內。
- `colab_tf.ipynb` 的 `6d`、`2d` 是競賽二較早版本的 TF；各 notebook 另有早期實驗或除錯用的 MODE，說明見各 notebook 開頭。
- `aicup.export_questions` 產生 val648 的固定評估題，`tf_run.py` 的 `pool`、`cv` 模式要把 `questions-s{47..51}.npz` 放在 `--data` 目錄。

| 報告「陸、分析與結論」的實驗 | 程式 |
|---|---|
| 加大序列模型容量（PV6.2） | `colab_seq_emb.ipynb` 的 `ARM = 'patch_conv_wide'` |
| 本地模擬題（PV2） | `aicup/eval_player_sim.py` |

## 授權

- [MIT License](LICENSE) 適用於本隊撰寫的程式碼，包括 `aicup/`、`unit_tests/`、`colab_*.ipynb`、`justfile`。
- `network.py`、`utils.py` 與兩份教學 notebook 來自官方教學 repository [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)，不在 MIT License 範圍內；該 repository 沒有 LICENSE 檔，其 README 聲明參賽者可自由使用、修改與散布其程式碼。
- 本專案使用 [KataGo](https://github.com/lightvector/KataGo)（MIT）與 Human SL 模型（[KataGo Neural Network License](https://katagotraining.org/network_license/)）。
- 訓練集僅供 2026 AI CUP 競賽使用，不得用於識別真實使用者。
