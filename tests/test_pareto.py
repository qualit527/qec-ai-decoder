def test_pareto_keeps_dominant_points():
    from autoqec.pareto.front import update_front
    front = []
    candidates = [
        {"id": "a", "delta_ler": 1e-4, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"},
        {"id": "b", "delta_ler": 5e-5, "flops": 5_000,  "n_params": 10_000, "verdict": "VERIFIED"},
        {"id": "c", "delta_ler": 2e-4, "flops": 50_000, "n_params": 100_000, "verdict": "VERIFIED"},
        {"id": "d", "delta_ler": 1e-5, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"},  # dominated by a
    ]
    for c in candidates:
        front = update_front(front, c)
    ids = {x["id"] for x in front}
    assert "d" not in ids
    assert ids == {"a", "b", "c"}


def test_pareto_skips_unverified():
    from autoqec.pareto.front import update_front
    front = update_front([], {"id": "x", "delta_ler": 1e-3, "flops": 100,
                                "n_params": 100, "verdict": "FAILED"})
    assert front == []


def test_pareto_dedup():
    from autoqec.pareto.front import update_front
    c = {"id": "a", "delta_ler": 1e-4, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"}
    front = update_front([], c)
    c2 = {"id": "a2", "delta_ler": 1e-4, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"}
    front = update_front(front, c2)
    assert len(front) == 1


def test_pareto_sorted_by_delta():
    from autoqec.pareto.front import update_front
    candidates = [
        {"id": "b", "delta_ler": 5e-5, "flops": 5_000, "n_params": 10_000, "verdict": "VERIFIED"},
        {"id": "c", "delta_ler": 2e-4, "flops": 50_000, "n_params": 100_000, "verdict": "VERIFIED"},
        {"id": "a", "delta_ler": 1e-4, "flops": 10_000, "n_params": 20_000, "verdict": "VERIFIED"},
    ]
    front = []
    for c in candidates:
        front = update_front(front, c)
    # Sorted by -delta_ler → c (2e-4) first
    assert front[0]["id"] == "c"
    deltas = [x["delta_ler"] for x in front]
    assert deltas == sorted(deltas, reverse=True)
