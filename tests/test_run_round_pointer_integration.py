import json
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

from autoqec.envs.schema import load_env_yaml
from autoqec.orchestration.subprocess_runner import run_round_in_subprocess
from autoqec.orchestration.worktree import cleanup_round_worktree, create_round_worktree
from autoqec.runner.schema import RunnerConfig


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.integration
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="subprocess runner invokes torch via run-round")
def test_run_round_commits_and_writes_pointer(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    wt = None
    try:
        wt = create_round_worktree(
            repo_root=repo,
            run_id="20260423-000000",
            round_idx=1,
            slug="smoke",
            fork_from="HEAD",
        )
        round_dir = tmp_path / "runs" / "20260423-000000" / "round_1"
        cfg = RunnerConfig(
            env_name="surface_d5_depol",
            predecoder_config=yaml.safe_load(
                (repo / "autoqec/example_db/gnn_small.yaml").read_text(encoding="utf-8")
            ),
            training_profile="dev",
            seed=0,
            round_dir=str(round_dir),
            code_cwd=wt["worktree_dir"],
            branch=wt["branch"],
            round_attempt_id="test-uuid-1",
        )
        env = load_env_yaml(repo / "autoqec/envs/builtin/surface_d5_depol.yaml")
        metrics = run_round_in_subprocess(cfg, env, round_attempt_id="test-uuid-1")

        assert metrics.status == "ok"
        assert metrics.commit_sha is not None
        pointer_path = Path(wt["worktree_dir"]) / "round_1" / "round_1_pointer.json"
        assert pointer_path.exists()

        p = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert p["round_attempt_id"] == "test-uuid-1"
        assert p["branch"] == wt["branch"]

        tip = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", wt["branch"]]
        ).decode().strip()
        assert tip == metrics.commit_sha
        shown = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "show",
                f"{wt['branch']}:round_1/round_1_pointer.json",
            ],
            text=True,
        )
        assert json.loads(shown)["round_attempt_id"] == "test-uuid-1"
    finally:
        if wt is not None:
            cleanup_round_worktree(repo, wt["worktree_dir"])
            subprocess.run(
                ["git", "-C", str(repo), "branch", "-D", wt["branch"]],
                capture_output=True,
                text=True,
            )
