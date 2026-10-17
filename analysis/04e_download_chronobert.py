"""Step 4e (extension E3): download the ChronoBERT vintages used by 04f (about 600 MB each, about 15 GB in total).

Vintages 1999-12-31 .. 2024-12-31 (meetings in year Y use vintage Y-1; 2024-12-31 is used for M5f).
Files are stored as plain copies (no symlinks; Windows without developer mode cannot create them) in
models_cache/chronobert/<vintage>/, which is gitignored. Delete that folder to reclaim the disk space.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import MODELS_CACHE  # noqa: E402

VINTAGES = [f"{y}1231" for y in range(1999, 2025)]
DEST = MODELS_CACHE / "chronobert"


def local_path(vintage: str) -> Path:
    return DEST / vintage


def main() -> None:
    for v in VINTAGES:
        d = local_path(v)
        if (d / "model.safetensors").exists():
            print("have", v, flush=True)
            continue
        snapshot_download(f"manelalab/chrono-bert-v1-{v}", local_dir=d,
                          allow_patterns=["*.json", "*.safetensors", "*.txt"])
        print("ok", v, flush=True)
    print("all vintages downloaded")


if __name__ == "__main__":
    main()
