"""Regression: cost_trace must reflect cumulative spend after each rung."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs import AMFRSPipeline, AMFRSRunConfig


def test_cost_trace_monotonic_across_rungs(tmp_path):
    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "out"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "out")
    cfg.generations = 2

    artifacts = AMFRSPipeline(cfg).run()
    costs = [float(e["spent_cost"]) for e in artifacts.cost_trace]
    assert costs, "expected cost_trace entries"
    for prev, nxt in zip(costs, costs[1:]):
        assert nxt >= prev, (prev, nxt)


def test_same_candidate_later_rung_has_higher_or_equal_cost(tmp_path):
    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "out"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "out")
    cfg.generations = 1

    artifacts = AMFRSPipeline(cfg).run()
    by_cand: dict[str, list[tuple[str, float]]] = {}
    for row in artifacts.cost_trace:
        cid = str(row["candidate_id"])
        by_cand.setdefault(cid, []).append((str(row["level"]), float(row["spent_cost"])))

    saw_increase = False
    for entries in by_cand.values():
        for (_lv_a, c_a), (lv_b, c_b) in zip(entries, entries[1:]):
            assert c_b >= c_a, (entries, lv_b)
            if c_b > c_a:
                saw_increase = True
    assert saw_increase, "expected at least one rung transition to increase spent_cost"
