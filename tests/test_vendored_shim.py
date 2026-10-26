"""The vendored safetensors shim must stay a byte-for-byte copy of the pipeline repository's canonical file.

Canonical: kurtvalcorza/tabicl-classifier-pipeline, src/tabicl_classifier_pipeline/safetensors_shim.py.

To re-sync after the canonical file changes:
1. in tabicl-classifier-pipeline run ``python tools/print_shim_sha256.py`` (CI prints it too);
2. copy that file over ``tabicl_safetensors_shim.py`` here, unchanged (LF line endings);
3. set ``PINNED_SHA256`` below to the printed digest.
A digest mismatch here means the copy drifted or was edited in place: fix it by re-syncing, never by editing the copy.
"""

import hashlib
from pathlib import Path

PINNED_SHA256 = "052189050681500cd327de0362f81886cb1ab138b2647027e2e14caaa04c7653"
VENDORED = Path(__file__).resolve().parents[1] / "tabicl_safetensors_shim.py"


def test_vendored_shim_is_the_pinned_byte_copy():
    data = VENDORED.read_bytes()
    assert b"\r\n" not in data, "the pin is on the LF bytes"
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256, "vendored shim drifted from the pinned canonical copy; re-sync it (see this file's docstring)"


def test_vendored_shim_header_names_its_canonical_source():
    text = VENDORED.read_text(encoding="utf-8")
    assert "CANONICAL SOURCE: kurtvalcorza/tabicl-classifier-pipeline, src/tabicl_classifier_pipeline/safetensors_shim.py." in text
    assert "kurtvalcorza/tabicl-classifier-finetuner as\ntabicl_safetensors_shim.py" in text


def test_vendored_shim_imports_without_tabicl_or_torch():
    import tabicl_safetensors_shim as shim

    assert shim.ARTIFACT_FORMAT == "tabicl-dimer-classifier-v2" and shim.BUNDLE_CHECKPOINT == "checkpoints/best.safetensors"
    assert shim.SUPPORTED_ARTIFACT_FORMATS == ("tabicl-dimer-classifier-v2", "tabicl-dimer-classifier-v1")
