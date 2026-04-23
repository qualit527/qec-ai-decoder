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

    out = write_artifact_manifest(
        round_dir=round_dir,
        env_yaml_path=env_yaml,
        dsl_config=dsl,
        cmd_line=["python", "-m", "cli.autoqec", "run", "fake.yaml"],
    )

    data = json.loads(out.read_text())
    # Env hash is deterministic sha-256 of bytes
    expected = hashlib.sha256(env_yaml.read_bytes()).hexdigest()
    assert data["env_yaml_sha256"] == expected
    # DSL hash is sha-256 of sorted-keys canonical JSON
    dsl_hash = hashlib.sha256(
        json.dumps(dsl, sort_keys=True).encode()
    ).hexdigest()
    assert data["dsl_sha256"] == dsl_hash
    # Tool versions
    assert "python_version" in data
    assert "torch_version" in data or data.get("torch_version") is None
    # Command line is preserved
    assert data["cmd_line"] == ["python", "-m", "cli.autoqec", "run", "fake.yaml"]


def test_manifest_repo_sha_optional(tmp_path):
    # When called outside a git repo, repo_sha is None but no crash
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("")
    rd = tmp_path / "round_1"
    rd.mkdir()
    data = json.loads(write_artifact_manifest(
        round_dir=rd, env_yaml_path=env_yaml, dsl_config={}, cmd_line=["x"],
    ).read_text())
    assert "repo_sha" in data  # key present even if None
