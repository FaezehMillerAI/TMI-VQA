import os
import shutil
import tempfile
import pytest
import torch

from cigci_eval.inference_adapter import run_dataset_inference
from cigci_eval.records import read_run


@pytest.fixture
def temp_output_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_inference_adapter_slake_sample(temp_output_dir):
    out_file = run_dataset_inference(
        dataset_name="slake",
        split="test",
        model_name="ci_gci",
        seed=42,
        device="cpu",
        output_dir=temp_output_dir,
        limit_samples=2,
    )
    assert os.path.exists(out_file)
    records, manifest = read_run(out_file)
    assert len(records) == 2
    assert manifest.model == "ci_gci"
    assert manifest.dataset == "slake"
    for r in records:
        assert 0.0 <= r.confidence <= 1.0
        assert abs(sum(r.prob_vector) - 1.0) < 1e-4
        assert r.logits_cf is not None


def test_inference_adapter_ablation_modes(temp_output_dir):
    # Test black_box and blur ablation modes
    for mode in ["black_box", "blur", "nearest"]:
        out_file = run_dataset_inference(
            dataset_name="slake",
            split="test",
            model_name=f"ablation_{mode}",
            inpainter_mode=mode,
            seed=42,
            device="cpu",
            output_dir=temp_output_dir,
            limit_samples=2,
        )
        assert os.path.exists(out_file)
        records, manifest = read_run(out_file)
        assert len(records) == 2
        assert manifest.model == f"ablation_{mode}"
