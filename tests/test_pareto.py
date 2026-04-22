def test_pareto_keeps_dominant_points():
    from autoqec.pareto.front import update_front, is_pareto_dominated
    front = []
    # (delta_ler, flops, n_params) — higher delta, lower flops+params better
    candidates = [
        {"id": "a", "delta_ler": 1e-4, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"},
        {"id": "b", "delta_ler": 5e-5, "flops": 5_000,  "n_params": 10_000, "verdict": "VERIFIED"},
        {"id": "c", "delta_ler": 2e-4, "flops": 50_000, "n_params": 100_000, "verdict": "VERIFIED"},
        {"id": "d", "delta_ler": 1e-5, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"},  # dominated by a
    ]
    for c in candidates:
        front = update_front(front, c)
    ids = {x["id"] for x in front}
    assert "d" not in ids       # dominated
    assert ids == {"a", "b", "c"}

def test_pareto_skips_unverified():
    from autoqec.pareto.front import update_front
    front = update_front([], {"id": "x", "delta_ler": 1e-3, "flops": 100,
                                "n_params": 100, "verdict": "FAILED"})
    assert front == []
