# 執行競賽一、競賽二的本機步驟，just --list 列出全部 recipe

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

hsl := "humansl-analysis"
pm := "player-trained-models"

# 列出 recipe
default:
    @just --list --unsorted

# 單元測試
test:
    pytest

# 玩家切分清單，已存在時略過
splits:
    {{ if path_exists(hsl / "train-p4.csv") == "true" { "echo 'skip: split lists exist'" } else { "python -m aicup.val_players --val-size 180000 --out-dir " + hsl } }}

# 6d profile 的 Human SL 分析局，已存在時略過
games6d: splits
    {{ if path_exists(hsl / "hist6d-newdev/games.csv") == "true" { "echo 'skip: hist6d games exist'" } else { "python -m aicup.hsl_games --run trainpool val648 newdev" } }}

# 競賽一第 1 步：本機資料準備
task1-prep: games6d
    python -m aicup.extract_features --train-dir training-v1.1/training
    python -m aicup.extract_features --train-dir training-v1.1/training --val-size 180000 --out rank-features-v180000
    python -m aicup.color --train-dir training-v1.1/training --out rank-features-v180000/hd.npz
    python -m aicup.prep_player_hsl --tests-dir tests
    python -m aicup.pack_move_seq --state-feats --out-dir move-seq-state-v180000
    python -m aicup.pack_move_seq --state-feats --extra-split hold={{hsl}}/holdout-v180000.csv --splits hold \
      --pat3-vocab move-seq-state-v180000/pat3_vocab.npy --out-dir move-seq-state-v180000
    python -m aicup.pack_move_patch --seq-dir move-seq-state-v180000 --out-dir move-seq-patch-v180000
    python -m aicup.pack_move_patch --seq-dir move-seq-state-v180000 --out-dir move-seq-patch-v180000 --splits hold

# 競賽一第 3 步：本機推論與融合，寫出 submission_player.csv
task1-submit:
    python -m aicup.infer_player_rank --ckpt rank-trained-models/rank-cnn/best_model.pth
    python -m aicup.infer_player_hsl
    python -m aicup.fuse_player --color-cond --pattern-beta 0.7 \
      --cand2-cache {{pm}}/hsl-cache/cand2_hsl_ranks.json \
      --query2-cache {{pm}}/hsl-cache/query2_hsl_ranks.json \
      --emb-gamma 1.0 --csls-k 5 --emb-top2-weight 0.5 --raw-open \
      --emb-cache-dir {{pm}}/emb-model-b/emb-cache \
      --seq-cache-dir {{pm}}/seq-state/emb-cache {{pm}}/seq-state_conf/emb-cache \
        {{pm}}/seq-patch/emb-cache {{pm}}/seq-patch_conv/emb-cache {{pm}}/seq-patch_crop/emb-cache \
        {{pm}}/seq-patch_d4/emb-cache {{pm}}/seq-patch_d4_val648/emb-cache \
      --seq-groups all --seq-weights 0.5 0.5 0.1 0.1 0.1 0.1 0.1 \
      --pub2-alpha 0.5 --pub2-emb-gamma 0.6 --pub2-open-weight 0.0 \
      --pub2-seq-weights 0.3 0.3 0.4 0.4 0.4 0.4 0.4 \
      --out-csv submission_player.csv

# 競賽二第 1 步：本機抽局
task2-games: games6d
    python -m aicup.hsl_games --2d --run val648 train_p1 train_p2 train_p3 train_p4 train_p4b

# 競賽二第 3 步：本機打包
task2-pack:
    python -m aicup.pack_moves --6d
    python -m aicup.pack_moves val648 train_p1 train_p2 test train_p3 train_p4all

# 競賽二第 4 步：A、B 模型
task2-ab: (concat hsl / "dev6d/dev-p3-hist.jsonl" hsl / "hist6d-val648/val648-p3-hist.jsonl" hsl / "hist6d-newdev/newdev-p3-hist.jsonl") (concat hsl / "dev6d/dev-p3-opp-hist.jsonl" hsl / "hist6d-val648/val648-p3-opp-hist.jsonl" hsl / "hist6d-newdev/newdev-p3-opp-hist.jsonl") (concat hsl / "hist2d-train_p12/train_p12-p3-hist.jsonl" hsl / "hist2d-train_p1/train_p1-p3-hist.jsonl" hsl / "hist2d-train_p2/train_p2-p3-hist.jsonl") (concat hsl / "hist2d-train_p12/train_p12-p3-opp-hist.jsonl" hsl / "hist2d-train_p1/train_p1-p3-opp-hist.jsonl" hsl / "hist2d-train_p2/train_p2-p3-opp-hist.jsonl")
    python -m aicup.predict_pool --s1-only --same-color --profiles rank_11k,rank_1d,rank_6d \
      --val-jsonl {{hsl}}/dev6d/dev-p3-hist.jsonl --val-opp {{hsl}}/dev6d/dev-p3-opp-hist.jsonl \
      --pool-jsonl {{hsl}}/hist6d-trainpool/trainpool-p3-hist.jsonl --pool-opp {{hsl}}/hist6d-trainpool/trainpool-p3-opp-hist.jsonl \
      --test-jsonl {{hsl}}/hist6d-test/test-p3-hist.jsonl --test-opp {{hsl}}/hist6d-test/test-p3-opp-hist.jsonl \
      --test-dir tests --save-probs rank-trained-models/task2-a/probs.npz --out-dir rank-trained-models/task2-a
    python -m aicup.predict_pool --s1-only --same-color --loss \
      --val-jsonl {{hsl}}/hist2d-val648/val648-p3-hist.jsonl --val-opp {{hsl}}/hist2d-val648/val648-p3-opp-hist.jsonl \
      --pool-jsonl {{hsl}}/hist2d-train_p12/train_p12-p3-hist.jsonl --pool-opp {{hsl}}/hist2d-train_p12/train_p12-p3-opp-hist.jsonl \
      --exclude-players {{hsl}}/val-players-dev-v180000.csv {{hsl}}/holdout-v180000.csv {{hsl}}/train-pool-v180000.csv \
      --test-jsonl {{hsl}}/hist2d-test/test-p3-hist.jsonl --test-opp {{hsl}}/hist2d-test/test-p3-opp-hist.jsonl \
      --test-dir tests --save-probs rank-trained-models/task2-b/probs.npz --out-dir rank-trained-models/task2-b

# 競賽二第 6 步：融合 A、B、TF，寫出提交檔
task2-submit:
    python -m aicup.blend_submit --a rank-trained-models/task2-a/probs.npz --b rank-trained-models/task2-b/probs.npz \
      --tf-dir {{hsl}}/tf-final-j30-p4all/out --out-dir rank-trained-models/fusion

# 串接 jsonl，輸出已存在時略過
[private]
concat out +inputs:
    {{ if path_exists(out) == "true" { "echo 'skip: " + out + " exists'" } else { "python -m aicup.concat_jsonl --out " + out + " " + inputs } }}
