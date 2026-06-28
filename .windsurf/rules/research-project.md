---
trigger: always_on
description: "General research project rules — TODO/FIXME, breadth-first methodology, documentation, code hygiene"
---

# Research Project Rules

## TODO / FIXME Comments
- Add `//TODO(reason):` or `#TODO(reason):` for incomplete items
- Add `//FIXME(reason):` or `#FIXME(reason):` for known bugs or issues
- Add `//HACK(reason):` or `#HACK(reason):` for temporary workarounds
- Add `//NOTE(reason):` or `#NOTE(reason):` for important context
- Every TODO/FIXME must include a reason in parentheses
- Never remove TODO/FIXME comments unless the item is completed

## Code Hygiene
- Remove unused imports, variables, and dead code
- No deprecated/re-export shim files — delete and update imports directly
- One class/concept per file
- Keep functions under 100 lines
- Follow existing naming conventions in the project

## Research Methodology
- ALWAYS explore ALL approaches for every design decision (breadth-first)
- NEVER commit to a single approach without testing alternatives
- Every module should have a corresponding test
- Findings must be documented after each experiment
- Literature must be surveyed before implementing
- Search broadly: check adjacent fields, not just the obvious ones
- Document what doesn't work, not just what does
- Prefer minimal upstream fixes over downstream workarounds
- Identify root cause before implementing fixes

## Documentation
- Maintain 3 core docs: README, TODO, CHANGELOG
- Update TODO when tasks are completed
- Update CHANGELOG when changes are made
- Keep docs in a `docs/` folder
- Max 3 `.md`/`.txt` files per update cycle

## Code Quality Checklist
- No syntax/logic errors?
- Imports present?
- Follows existing style?
- No dead code?
- No hardcoded secrets?
- Clear errors?
- No vulnerabilities?

## Error Handling
- Handle all errors, no bare try/catch
- Fail safely with clear messages and context
- Validate input
- No silent failures

## Testing
- Design or update tests before major implementation work
- Never delete or weaken tests without explicit direction
- Think edge cases early: what test catches this bug?
- Don't mock what you don't own

## Performance
- Avoid nested loops, flag O(n²)
- Cache repeats, no premature optimization
- Watch concurrency issues

## Dependencies
- Pin versions
- Prefer maintained packages
- Check size/license/alternatives first
- Remove unused
