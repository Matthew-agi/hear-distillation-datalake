"""Streaming datasets shared by the training objectives."""

from .shards import AudioShardDataset, discover_shards, iter_tar_pairs

__all__ = ["AudioShardDataset", "discover_shards", "iter_tar_pairs"]
