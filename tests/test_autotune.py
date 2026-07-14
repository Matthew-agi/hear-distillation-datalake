from hear_distill.autotune import HostResources, build_runtime_plan


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
    assert plan.train_batch_size == 128
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
