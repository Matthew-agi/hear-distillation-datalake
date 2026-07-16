from hear_distill.autotune import HostResources, build_runtime_plan, training_batch_size
from hear_distill.cli import _train_defaults
from hear_distill.models.memory import estimate_training_memory, rounded_initial_batch


def test_plan_uses_at_most_half_of_disk_and_preserves_free_space() -> None:
    resources = HostResources(
        cpu_count=32,
        disk_total_gib=1_000.0,
        disk_free_gib=800.0,
        gpu_name="NVIDIA H100",
        gpu_memory_gib=80.0,
    )
    plan = build_runtime_plan(resources)

    assert plan.disk_budget_gib == 500.0
    assert plan.lake_max_gib < plan.disk_budget_gib
    assert plan.disk_min_free_gib == 50.0
    assert plan.num_streams == 8
    assert plan.shuffle_buffer == 1_000
    assert plan.train_batch_size == training_batch_size(80.0)
    assert plan.amp is True
    assert plan.teacher_batch_factor == 2


def test_plan_is_limited_by_current_free_space() -> None:
    resources = HostResources(
        cpu_count=8,
        disk_total_gib=100.0,
        disk_free_gib=20.0,
        gpu_name=None,
        gpu_memory_gib=0.0,
    )
    plan = build_runtime_plan(resources)

    assert plan.disk_budget_gib == 15.0
    assert plan.device == "cpu"
    assert plan.amp is False
    assert plan.num_streams == 2
    assert plan.train_num_workers >= 1


def test_batch_plan_scales_with_model_and_objective() -> None:
    resources = HostResources(
        cpu_count=32,
        disk_total_gib=1_000.0,
        disk_free_gib=800.0,
        gpu_name="NVIDIA H100",
        gpu_memory_gib=80.0,
    )
    small_distill = build_runtime_plan(resources, model_size="small", objective="distill")
    base_reconstruct = build_runtime_plan(
        resources, model_size="base", objective="reconstruct"
    )
    large_reconstruct = build_runtime_plan(
        resources, model_size="large", objective="reconstruct"
    )

    assert small_distill.train_batch_size > base_reconstruct.train_batch_size
    assert base_reconstruct.train_batch_size > large_reconstruct.train_batch_size
    for plan, size, objective in (
        (small_distill, "small", "distill"),
        (base_reconstruct, "base", "reconstruct"),
        (large_reconstruct, "large", "reconstruct"),
    ):
        raw_cap = estimate_training_memory(size, objective).maximum_batch_size(80.0)
        assert 0 < plan.train_batch_size <= raw_cap
        assert plan.train_batch_size % 8 == 0
    assert base_reconstruct.teacher_batch_factor == 1


def test_adaptive_warmup_starts_at_largest_supported_power_of_two() -> None:
    assert rounded_initial_batch(224) == 128
    assert rounded_initial_batch(128) == 128
    assert rounded_initial_batch(7) == 4
    distill_defaults = _train_defaults(
        objective="distill",
        model_size="small",
        device="cuda",
        amp=True,
        teacher_batch_factor=1,
        batch_cap=224,
        max_steps=10_000,
    )
    probe_index = distill_defaults.index("--auto-warmup-probe-batch-size")
    assert distill_defaults[probe_index + 1] == "128"
