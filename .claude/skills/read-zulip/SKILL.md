---
name: read-zulip
description: Read Zulip stream/topic history for project context, summarize decisions, and recover requirements or action items. Use when task context may depend on prior Zulip discussion rather than only repository files.
---

# /read-zulip

Read Zulip history with shared credentials, then turn the result into project
context for the current task.

## When to use

- A task references hackathon discussion, prior design debate, or off-repo
  decisions that probably live in Zulip.
- Repository docs are insufficient and you need the original chat context.
- The user asks you to inspect a specific Zulip stream, topic, or channel URL.

## Auth resolution

Use `scripts/read_zulip.py`. It resolves credentials in this order:

1. `ZULIPRC`
2. `~/.zuliprc`
3. `~/.config/zulip/zuliprc`

If a collaborator keeps credentials elsewhere, pass `--config /path/to/zuliprc`
explicitly.

Never commit Zulip credentials into this repository.

## Repo defaults

For this repository, start with:

```bash
python .claude/skills/read-zulip/scripts/read_zulip.py \
  --url "https://quantum-info.zulipchat.com/#channels/591576/VibeYoga-Hackathon-QEC/general" \
  --limit 20
```

If `general` returns no relevant messages, widen to the whole stream:

```bash
python .claude/skills/read-zulip/scripts/read_zulip.py \
  --channel 591576 \
  --limit 20
```

If you still need the active thread referenced by repo docs, check:

```bash
python .claude/skills/read-zulip/scripts/read_zulip.py \
  --channel 591576 \
  --topic "channel events" \
  --limit 20
```

## Workflow

1. Start with the narrowest stream/topic or URL that answers the question.
2. Fetch a small window first, usually `--limit 20`.
3. Expand only if the first pass is insufficient.
4. Summarize decisions, requirements, blockers, and action items with message
   IDs when they matter.

## Useful commands

Recent messages from a stream/topic:

```bash
python .claude/skills/read-zulip/scripts/read_zulip.py \
  --channel "VibeYoga-Hackathon-QEC" \
  --topic "channel events" \
  --limit 20
```

Local substring filter over the fetched window:

```bash
python .claude/skills/read-zulip/scripts/read_zulip.py \
  --channel 591576 \
  --limit 100 \
  --contains benchmark
```

Machine-readable output:

```bash
python .claude/skills/read-zulip/scripts/read_zulip.py \
  --channel 591576 \
  --limit 20 \
  --format json
```

## Output rules

- Default to plain-text output when you want to read and summarize.
- Use `--format json` only when another tool/script will consume the result.
- Keep summaries chronological when the discussion records changing decisions.
- Quote only short snippets when direct wording matters.
