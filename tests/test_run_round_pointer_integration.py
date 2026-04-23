import json
import subprocess
from pathlib import Path

import pytest

from autoqec.orchestration.worktree import create_round_worktree
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig


@pytest.mark.integration
def test_run_round_commits_and_writes_pointer(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-b", "main", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "init"])

    wt = create_round_worktree(
        repo_root=str(repo), run_id="20260423-000000", round_idx=1, slug="smoke",
    )
    round_dir = Path(wt["worktree_path"]) / "runs" / "20260423-000000" / "round_1"
    round_dir.mkdir(parents=True)

    cfg = RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={
            "type": "gnn", "output_mode": "soft_priors",
            "hidden_dim": 4, "n_layers": 1,
        },
        training_profile="dev",
        seed=0,
        round_dir=str(round_dir),
        code_cwd=wt["worktree_path"],
        branch=wt["branch"],
        round_attempt_id="test-uuid-1",
        commit_message="feat(test): smoke round",
    )
    metrics = run_round(cfg)

    assert metrics.status == "ok"
    assert metrics.commit_sha is not None
    assert (round_dir / "round_1_pointer.json").exists()

    p = json.loads((round_dir / "round_1_pointer.json").read_text())
    assert p["commit_sha"] == metrics.commit_sha

    tip = subprocess.check_output(
        ["git", "-C", wt["worktree_path"], "rev-parse", wt["branch"]]
    ).decode().strip()
    assert tip == metrics.commit_sha
