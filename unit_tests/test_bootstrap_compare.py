import numpy as np
import pytest

from aicup import bootstrap_compare as bc
from aicup import eval_posthoc
from aicup.scoring import LEVELS, apply_temperature


def _data(n_players=8, games_each=12, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    pid, rank = [], []
    for p in range(n_players):
        pid += ['p%d' % p] * games_each
        rank += [p % len(LEVELS)] * games_each
    rank = np.array(rank, np.int8)
    logits = rng.normal(size=(len(rank), len(LEVELS)))
    logits[np.arange(len(rank)), rank] += signal
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return {'probs': probs.astype(np.float32), 'rank_idx': rank,
            'key': np.array(['g%03d' % i for i in range(len(rank))]), 'player_id': np.array(pid)}


def test_align_reorders_b_to_a():
    a = _data()
    order = np.random.default_rng(1).permutation(len(a['key']))
    b = {k: v[order] for k, v in a.items()}
    out = bc.align(a, b)
    assert (out['key'] == a['key']).all()
    assert np.allclose(out['probs'], a['probs'])


def test_align_rejects_different_games():
    a = _data()
    b = _data(n_players=7)
    with pytest.raises(ValueError):
        bc.align(a, b)


def test_predict_cut_matches_eval_posthoc():
    d = _data()
    probs = d['probs'].astype(np.float64)
    logp = np.log(np.clip(probs, bc.EPS, None))
    questions, _ = eval_posthoc.make_questions(d['player_id'], d['rank_idx'], (5, 7), np.random.default_rng(0))
    flat = np.concatenate(questions)
    starts = np.cumsum([0] + [len(q) for q in questions[:-1]])
    for agg in ('mean', 'log'):
        for select in ('argmax', 'pick'):
            expected = eval_posthoc.predict_questions(probs, questions, agg, select)
            assert (bc.predict_cut(probs, logp, flat, starts, agg, select) == expected).all()


def test_identical_models_have_zero_diff():
    d = _data()
    res = bc.paired_bootstrap(d['probs'], d['probs'], d['player_id'], d['rank_idx'], 20, (5, 7), 'log', 'pick', 0)
    for key in ('single', 'questions'):
        s = bc.summarize(*res[key])
        assert s['diff'] == 0 and s['sd'] == 0


def test_better_model_has_positive_diff():
    weak = _data(signal=0.2)
    strong = _data(signal=3.0)
    res = bc.paired_bootstrap(weak['probs'], strong['probs'], weak['player_id'], weak['rank_idx'],
                              30, (5, 7), 'log', 'pick', 0)
    for key in ('single', 'questions'):
        s = bc.summarize(*res[key])
        assert s['lo'] > 0 and s['p_pos'] == 1.0


def test_tail_mean(tmp_path):
    p = tmp_path / 'history.csv'
    p.write_text('epoch,val_score_pick\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n5,0.5\n', encoding='utf-8')
    assert bc.tail_mean(str(p)) == pytest.approx(0.35)
    with pytest.raises(ValueError):
        bc.tail_mean(str(p), n=6)


def test_temperature_matches_scaled_predict_questions():
    d = _data()
    a, b = d['probs'].astype(np.float64), _data(seed=1)['probs'].astype(np.float64)
    # 每位玩家只有 12 局且每題 12 局，題目固定為該玩家的全部局，不受隨機影響
    res = bc.paired_bootstrap(a, b, d['player_id'], d['rank_idx'], 1, (12,), 'log', 'pick', 0, temp=3.0)
    members, member_labels = bc.player_groups(d['player_id'], d['rank_idx'])
    rng = np.random.default_rng(0)
    draw = rng.integers(0, len(members), len(members))
    questions = [members[c] for c in draw]
    q_labels = member_labels[draw]
    for k, probs in enumerate((a, b)):
        preds = eval_posthoc.predict_questions(apply_temperature(probs, 3.0), questions, 'log', 'pick')
        assert res['questions'][k][0] == pytest.approx(bc.game_scores(q_labels, preds).mean())
