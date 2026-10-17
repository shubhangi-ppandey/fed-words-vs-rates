"""Step 2b: SHA-256 manifest of every raw input file (data/raw_manifest.json).

Several raw sources are live (FRED, Kenneth French) and can be revised. The small raw files are committed to the
repository, and this manifest lets anyone verify that a fresh download matches the data behind the paper.
The 15 MB of raw statement HTML is not committed; it is re-created by step 01 and covered by the manifest.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import RAW, ROOT  # noqa: E402

files = sorted(p for p in RAW.rglob("*") if p.is_file())
manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                                          "bytes": p.stat().st_size} for p in files}
(ROOT / "data" / "raw_manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"manifest: {len(manifest)} files, {sum(v['bytes'] for v in manifest.values()) / 1e6:.1f} MB")
