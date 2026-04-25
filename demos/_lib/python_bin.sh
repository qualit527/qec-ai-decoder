# Shared helper for demo bash launchers.
#
# `discover_demo_python <start_dir>` echoes the absolute path of the
# project's `.venv` interpreter if one is found by walking up from
# `<start_dir>`, otherwise falls back to `python3` then `python`. This
# matches the Python-side resolver in scripts/run_bb72_demo.py so Demo 1
# and Demo 4 pick up the project venv even when the caller's PATH points
# at a system Python that lacks torch.
#
# Usage:
#   HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   REPO_ROOT="$(cd "$HERE/../.." && pwd)"
#   source "$REPO_ROOT/demos/_lib/python_bin.sh"
#   PY="${PYTHON:-$(discover_demo_python "$REPO_ROOT")}"

discover_demo_python() {
    local d="$1"
    while [[ -n "$d" && "$d" != "/" && "$d" != "." ]]; do
        # -f rather than -x so Windows (where Python's os.chmod cannot set
        # Unix exec bits) still resolves a freshly-created .venv correctly.
        if [[ -f "$d/.venv/Scripts/python.exe" ]]; then
            printf '%s\n' "$d/.venv/Scripts/python.exe"
            return 0
        fi
        if [[ -f "$d/.venv/bin/python" ]]; then
            printf '%s\n' "$d/.venv/bin/python"
            return 0
        fi
        local parent
        parent="$(dirname "$d")"
        if [[ "$parent" == "$d" ]]; then
            break
        fi
        d="$parent"
    done
    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
    fi
    printf '%s\n' "python"
}
