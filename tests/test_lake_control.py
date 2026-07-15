from datalake.run_lake import (
    _claimed_bytes_from_inventory,
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
