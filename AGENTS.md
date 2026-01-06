# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with remote
```

## Completing an Issue

When you finish an issue:

1. **Commit your changes** - `jj commit -m "..."`
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Verify implementation against spec** - Use the Task tool to spawn a subagent that:
   - Re-reads the issue spec (`bd show <id>`)
   - Reviews your implementation against each requirement
   - Reports any missing or incomplete items
   - If gaps are found, address them before proceeding
4. **Close the issue** - `bd close <id>`
5. **File issues for remaining work** - Create issues for anything that needs follow-up

Do NOT push after individual issues - continue working on the next issue in the epic.

## Landing the Plane (Epic Completion)

**When completing an epic**, you MUST complete ALL steps below. An epic is NOT complete until `jj git push` succeeds.

**MANDATORY WORKFLOW:**

1. **Verify all issues in the epic are closed** - `bd epic show <id>`
2. **Run quality gates** - Tests, linters, builds for the full changeset
3. **Clean up** - Abandon empty changes, clean up bookmarks
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   jj git fetch
   bd sync
   jj git push
   jj log -r 'remote_bookmarks()'  # Verify remote is synced
   ```
5. **Close the epic** - `bd close <epic-id>`
6. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- An epic is NOT complete until `jj git push` succeeds
- NEVER stop before pushing when finishing an epic - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

