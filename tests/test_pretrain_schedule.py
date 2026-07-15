from __future__ import annotations

from argparse import Namespace

import pytest

from hear_distill.training.pretrain import _adaptive_lr_at_step, _lr_at_step
from distill_hear_vit_s_canon2d import _lr_multiplier


def _args(**overrides: object) -> Namespace:
    values = {
        "lr": 4.0e-6,
        "lr_schedule": "none",
        "lr_schedule_start_step": 0,
        "lr_warmup_steps": 0,
        "auto_warmup_steps": 1_000,
        "max_steps": 46_200,
        "lr_min_ratio": 0.1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_stable_schedule_is_constant() -> None:
    args = _args()

    assert _lr_at_step(args, 1) == pytest.approx(args.lr)
    assert _lr_at_step(args, args.max_steps) == pytest.approx(args.lr)
    assert _adaptive_lr_at_step(args, step=42_000, warmup_lr=5.0e-6) == pytest.approx(
        5.0e-6
    )


def test_resumed_decay_anchors_to_checkpoint_lr_and_phase_start() -> None:
    args = _args(
        lr_schedule="linear",
        lr_schedule_start_step=42_000,
    )

    assert _adaptive_lr_at_step(args, step=42_000, warmup_lr=5.0e-6) == pytest.approx(
        4.0e-6
    )
    assert _adaptive_lr_at_step(args, step=44_100, warmup_lr=5.0e-6) == pytest.approx(
        2.2e-6
    )
    assert _adaptive_lr_at_step(args, step=46_200, warmup_lr=5.0e-6) == pytest.approx(
        4.0e-7
    )


def test_distillation_linear_decay_honors_minimum_ratio() -> None:
    assert _lr_multiplier(
        step=0,
        max_steps=4_200,
        schedule="linear",
        warmup_steps=0,
        min_ratio=0.1,
    ) == pytest.approx(1.0)
    assert _lr_multiplier(
        step=2_100,
        max_steps=4_200,
        schedule="linear",
        warmup_steps=0,
        min_ratio=0.1,
    ) == pytest.approx(0.55)
    assert _lr_multiplier(
        step=4_200,
        max_steps=4_200,
        schedule="linear",
        warmup_steps=0,
        min_ratio=0.1,
    ) == pytest.approx(0.1)
