# KEREN Student V0.1 Training Notes

Training uses QLoRA against `Qwen/Qwen3-0.6B-Base`. Base weights remain frozen; only the PEFT adapter is learned.

Key safeguards:

- train only on the compiled 400-record Gold V0.1 dataset
- keep the locked 30-case benchmark out of training
- mask the user/prompt portion from the language-model loss so optimization targets the KEREN answer only
- use NF4 4-bit base loading with double quantization
- apply LoRA to Q/K/V/O and MLP projection layers
- save resumable checkpoints and keep the latest three
- record a training manifest next to the adapter
- never commit model weights, checkpoints, or benchmark outputs to GitHub

The canonical run procedure is `docs/training_run_v0.1.md`.
