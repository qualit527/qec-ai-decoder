#!/usr/bin/env bash
# Re-clone all autoresearch reference repos into this folder.
# Usage: bash clone_refs.sh [--shallow | --full]
# Default: --shallow (--depth 1, no history)

set -e
cd "$(dirname "$0")"

MODE="${1:---shallow}"
DEPTH_FLAG=""
[ "$MODE" = "--shallow" ] && DEPTH_FLAG="--depth 1"

# Format: target_dir|repo_url
REPOS=(
  "ai-scientist|https://github.com/SakanaAI/AI-Scientist.git"
  "ai-scientist-v2|https://github.com/SakanaAI/AI-Scientist-v2.git"
  "agent-laboratory|https://github.com/SamuelSchmidgall/AgentLaboratory.git"
  "aide|https://github.com/WecoAI/aideml.git"
  "karpathy-autoresearch|https://github.com/karpathy/autoresearch.git"
  "rd-agent|https://github.com/microsoft/RD-Agent.git"
  "problem-reductions|https://github.com/CodingThrust/problem-reductions.git"
)

for entry in "${REPOS[@]}"; do
  dir="${entry%%|*}"
  url="${entry##*|}"
  if [ -d "$dir/.git" ] || [ -d "$dir" ]; then
    echo "skip $dir (exists)"
    continue
  fi
  echo "cloning $url -> $dir"
  git clone --quiet $DEPTH_FLAG "$url" "$dir"
done

echo
echo "All clones done. Sizes:"
du -sh */ 2>/dev/null | sort -h
