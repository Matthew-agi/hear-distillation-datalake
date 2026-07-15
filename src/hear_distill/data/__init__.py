"""Streaming datasets shared by the training objectives."""

from .shards import AudioShardDataset, discard_claimed_shards, discover_shards, iter_tar_pairs

__all__ = ["AudioShardDataset", "discard_claimed_shards", "discover_shards", "iter_tar_pairs"]
