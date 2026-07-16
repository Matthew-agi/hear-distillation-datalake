import json
import tarfile
from pathlib import Path

import torch

from datalake.run_lake import (
    _claimed_bytes_from_inventory,
    _checkpoint_trainer_defaults,
    _load_curation_state,
)


def test_fresh_consumption_comes_from_claimed_inventory_not_batch_cap() -> None:
    assert (
        _claimed_bytes_from_inventory(
            origin_active_bytes=100,
            produced_since_origin_bytes=80,
            active_bytes=130,
        )
        == 50
    )


def test_fresh_consumption_is_monotonic_across_inventory_refreshes() -> None:
    assert (
        _claimed_bytes_from_inventory(
            origin_active_bytes=100,
            produced_since_origin_bytes=80,
            active_bytes=140,
            previous_claimed_bytes=50,
        )
        == 50
    )


def test_decay_resume_preserves_completed_warmup_configuration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "args": {
                "batch_size": 224,
                "grad_accum": 1,
                "num_workers": 10,
                "lr": 3.0e-4,
            },
            "adaptive_warmup_state": {"config": {"max_lr": 3.0e-4}},
        },
        checkpoint,
    )

    defaults = _checkpoint_trainer_defaults(checkpoint)

    assert defaults["batch_size"] == 224
    assert defaults["original_lr"] == 3.0e-4
    assert defaults["auto_warmup_max_lr"] == 3.0e-4


def test_legacy_copied_decay_store_is_discarded(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    decay_dir = tmp_path / "decay"
    for directory in (train_dir, val_dir, decay_dir):
        directory.mkdir()
    legacy_shard = decay_dir / "shard-000000.tar"
    with tarfile.open(legacy_shard, "w"):
        pass
    state_path = tmp_path / "curation.json"
    state_path.write_text(
        json.dumps(
            {
                "decay_active": [
                    {"tar_path": str(legacy_shard), "stem": "old", "nbytes": 1, "score": 0.1}
                ]
            }
        )
    )

    state = _load_curation_state(
        state_path,
        train_lake_dir=train_dir,
        val_lake_dir=val_dir,
        decay_lake_dir=decay_dir,
        num_streams=1,
        val_capacity_entries=0,
        decay_max_entries=10,
    )

    assert state["decay_split_mode"] == "exclusive_v1"
    assert state["decay_active"] == []
    assert state["discarded_legacy_decay_shards"] == 1
    assert not legacy_shard.exists()
