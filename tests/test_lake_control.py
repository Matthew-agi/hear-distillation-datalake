from datalake.run_lake import _logical_reserve_bytes, _prune_target_bytes


def test_reserve_is_relative_to_run_origin_after_resume() -> None:
    reserve = _logical_reserve_bytes(
        active_bytes=60,
        origin_active_bytes=50,
        produced_since_origin_bytes=20,
        consumed_since_origin_bytes=10,
    )
    assert reserve == 60


def test_reserve_never_claims_more_than_is_on_disk() -> None:
    reserve = _logical_reserve_bytes(
        active_bytes=12,
        origin_active_bytes=50,
        produced_since_origin_bytes=20,
        consumed_since_origin_bytes=10,
    )
    assert reserve == 12


def test_pruning_removes_only_reserve_above_low_water_mark() -> None:
    target = _prune_target_bytes(active_bytes=100, reserve_bytes=80, reserve_low_bytes=20)
    assert target == 40

    reserve_after_prune = _logical_reserve_bytes(
        active_bytes=target,
        origin_active_bytes=100,
        produced_since_origin_bytes=0,
        consumed_since_origin_bytes=20,
        pruned_since_origin_bytes=60,
    )
    assert reserve_after_prune == 20
