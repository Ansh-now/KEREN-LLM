"""Download the pinned KEREN V0.1 base model from Hugging Face.

Keep model weights outside Git. Example:
    python scripts/download_base_model.py D:/KEREN-Student/models/Qwen3-0.6B-Base
"""
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B-Base"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/download_base_model.py <output-directory>")
    output = Path(sys.argv[1]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(output),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {MODEL_ID} to {output}")


if __name__ == "__main__":
    main()
