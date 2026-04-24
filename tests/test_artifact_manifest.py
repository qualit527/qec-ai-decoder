import hashlib
import json

from autoqec.runner.manifest import write_artifact_manifest


def test_manifest_contains_repo_sha_and_hashes(tmp_path):
    # Arrange a fake env YAML + DSL config
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("name: test_env\n")
    dsl = {"type": "gnn", "hidden_dim": 4}
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")

    out = write_artifact_manifest(
        round_dir=round_dir,
        env_yaml_path=env_yaml,
        dsl_config=dsl,
        cmd_line=["python", "-m", "cli.autoqec", "run", "fake.yaml"],
    )

    data = json.loads(out.read_text())
    # Env hash is deterministic sha-256 of bytes
    expected = hashlib.sha256(env_yaml.read_bytes()).hexdigest()
    assert data["schema_version"] == 1
    assert data["environment"]["env_yaml_sha256"] == expected
    # DSL hash is sha-256 of sorted-keys canonical JSON
    dsl_hash = hashlib.sha256(
        json.dumps(dsl, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert data["round"]["dsl_config_sha256"] == dsl_hash
    # Tool versions
    assert "python" in data["packages"]
    assert "torch" in data["packages"]
    # Command line is preserved
    assert data["round"]["command_line"] == ["python", "-m", "cli.autoqec", "run", "fake.yaml"]


def test_manifest_repo_sha_optional(tmp_path):
    # When called outside a git repo, repo_sha is None but no crash
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("")
    rd = tmp_path / "round_1"
    rd.mkdir()
    (rd / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (rd / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (rd / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (rd / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    data = json.loads(write_artifact_manifest(
        round_dir=rd, env_yaml_path=env_yaml, dsl_config={}, cmd_line=["x"],
    ).read_text())
    assert "commit_sha" in data["repo"]  # key present even if None
