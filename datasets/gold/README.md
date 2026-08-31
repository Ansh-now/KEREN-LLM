# Gold Dataset V0.1

This directory contains the hand-designed, owner-reviewable foundation examples for KEREN Student V0.1.

## Target
First milestone: **50–100 high-quality records** before bulk generation.

Coverage must deliberately include:
- normal success
- ambiguity / clarification
- unsafe or confirmation-required actions
- unavailable/offline nodes
- internet unavailable
- incorrect or unavailable tools
- execution failure
- verification failure
- recovery/retry
- escalation
- Hinglish/English and noisy/STT-style inputs
- debugging/coding/embedded examples
- KEREN self-engineering examples

Gold records are not bulk paraphrases. Each example should teach a meaningful behavior or boundary.

## Rules
- Never store secrets or credentials.
- Do not fabricate execution evidence.
- Use reason codes instead of verbose hidden reasoning.
- Keep real KEREN contracts/tool names when known; mark provisional examples when contracts are not yet repo-grounded.
- Validation/test examples must be semantically separated from training variants to prevent paraphrase leakage.
