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
  "open-coscientist|https://github.com/conradry/open-coscientist-agents.git"
  "mlgym|https://github.com/facebookresearch/MLGym.git"
  "paper-qa|https://github.com/Future-House/paper-qa.git"
  "openevolve|https://github.com/codelion/openevolve.git"
  "ml-master|https://github.com/sjtu-sai-agents/ML-Master.git"
  "internagent|https://github.com/InternScience/InternAgent.git"
  "dolphin|https://github.com/InternScience/Dolphin.git"
  "sciagents|https://github.com/lamm-mit/SciAgentsDiscovery.git"
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
