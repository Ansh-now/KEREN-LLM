# KEREN Training Wheels Protocol V0.1

Training Wheels is an owner-controlled improvement pipeline. It must never silently retrain or replace production model weights.

## Separation of learning mechanisms

- **Memory** stores facts, context and preferences.
- **Procedure learning** stores reusable successful workflows.
- **Background enrichment** adds approved notes/examples/context.
- **Dataset collection** records sanitized future-training candidates.
- **Model training** modifies weights/adapters and requires a separate controlled training process.

## Candidate pipeline

Normal KEREN execution
→ capture candidate trace
→ remove credentials/secrets/PII
→ normalize
→ validate schema
→ deduplicate
→ score usefulness/quality
→ owner-controlled dataset staging
→ train candidate adapter/model
→ locked evaluation
→ safety/coding/task/latency regression checks
→ owner approval
→ production promotion
→ rollback available.

## Candidate trace fields

Useful traces can contain the request, relevant context, selected plan, actions, execution results, observations, verification, recovery, final result and explicit user correction/feedback. Sensitive raw data must not enter training storage.

## Authority rules

1. Collection does not imply approval for training.
2. Training does not imply approval for deployment.
3. Deployment requires benchmark/regression evidence.
4. Production model/adapter must be versioned and rollbackable.
5. High-risk authority decisions remain enforced by deterministic KEREN policy outside model weights.
6. Owner authorization should use trusted application/system controls, not a hidden natural-language phrase.

## Improvement levels

1. Memory optimization.
2. Procedure optimization.
3. Prompt/routing optimization.
4. LoRA/adapter training — owner-controlled.
5. Base/model update — owner-controlled + benchmark + rollback.
