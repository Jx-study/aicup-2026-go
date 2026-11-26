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

## 致謝與授權

- 官方教學 repository 由 NDHU AI Lab 的 Serkan Kavak 製作，`network.py`、`utils.py` 與兩份教學 notebook 來自該 repository。
- 官方教學 repository 允許參賽者自由使用、修改與散布其程式碼。
- 訓練集僅供 2026 AI CUP 競賽使用，不得用於識別真實使用者。
