import contextlib
from types import SimpleNamespace

import torch
import torch.nn as nn

import distill_hear_vit_s_canon2d as trainer
from adaptive_warmup import CriticalLREstimate


def test_critical_lr_probe_caches_preprocessing_and_teacher(monkeypatch) -> None:
    counts = {"preprocess": 0, "teacher": 0, "student": 0}

    def preprocess(batch: torch.Tensor) -> torch.Tensor:
        counts["preprocess"] += 1
        return batch

    class Teacher(nn.Module):
        def forward(self, spec: torch.Tensor, *, return_dict: bool):
            assert return_dict
            counts["teacher"] += 1
            return SimpleNamespace(pooler_output=torch.zeros_like(spec))

    class Student(nn.Linear):
        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            counts["student"] += 1
            return super().forward(spec)

    def fake_estimate(loss_closure, **_kwargs):
        base_loss = loss_closure()
        loss_closure()
        accepted_loss = loss_closure()
        return CriticalLREstimate(
            critical_lr=0.01,
            critical_sharpness=200.0,
            reference_lr=0.01,
            base_loss=base_loss,
            accepted_loss=accepted_loss,
            evaluations=3,
            bracketed=True,
        )

    monkeypatch.setattr(trainer, "estimate_critical_learning_rate", fake_estimate)
    student = Student(2, 2, bias=False)
    projection = nn.Identity()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01, weight_decay=0.0)
    student.weight.grad = torch.ones_like(student.weight)
    original_weight = student.weight.detach().clone()
    batches = [torch.ones(2, 2)]

    estimate, _ = trainer._estimate_critical_lr(
        loader=batches,
        data_iter=iter(batches),
        preprocess_audio=preprocess,
        teacher=Teacher(),
        student=student,
        proj=projection,
        optim=optimizer,
        device=torch.device("cpu"),
        contrastive_temp=0.1,
        loss_mse_weight=1.0,
        loss_contrastive_weight=0.0,
        loss_relational_weight=0.0,
        autocast_ctx=contextlib.nullcontext,
        teacher_autocast_ctx=contextlib.nullcontext,
        current_lr=0.01,
        prev_estimate=None,
        max_lr_cap=None,
    )

    assert estimate is not None
    assert estimate.critical_lr == 0.01
    assert estimate.critical_sharpness == 200.0
    assert counts == {"preprocess": 1, "teacher": 1, "student": 3}
    assert torch.allclose(student.weight, original_weight, rtol=0.0, atol=1e-7)
