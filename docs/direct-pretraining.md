# Direct Canon pretraining

## Why masked spectrogram reconstruction

The direct objective learns from the same unlabeled LAION-Audio clips as the
distillation pipeline without running the HeAR teacher. Masked spectrogram
modeling is established as a useful route to general-purpose audio
representations, and asymmetric encoder/decoder pretraining lets the decoder
be discarded after representation learning.

Relevant starting points:

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [Masked Autoencoders that Listen](https://arxiv.org/abs/2207.06405)
- [Masked Spectrogram Modeling using Masked Autoencoders](https://arxiv.org/abs/2204.12260)
- [MAE-AST](https://arxiv.org/abs/2203.16691)
- [SimMIM](https://arxiv.org/abs/2111.09886)

## Why this encoder keeps masked tokens

Canonical MAE removes masked tokens before its encoder. That is efficient, but
Canon2D needs a rectangular token set so it can reshape `[B, H*W, C]` into the
time-frequency grid for depthwise 2D convolution. A visible-token-only set has
no such dense grid.

This implementation therefore follows the simpler masked-input reconstruction
family: it replaces masked patch embeddings with one learned token and sends
the full grid through the encoder. For the current `[1, 192, 128]` input and
patch size 16, every Canon2D layer consistently sees `12 × 8` patches. The
loss is still computed only over masked targets.

## Coordinate-aware sparse Canon option

A future visible-token encoder can recover conventional MAE savings without
discarding Canon's coordinates. Attention and MLP operations would run only on
visible tokens. At each Canon placement, the implementation would scatter
those tokens into the known 12-by-8 coordinates, apply a mask-normalized
depthwise convolution, and gather the visible results again.

This is a moderate model refactor rather than a new research dependency. Block
calls must carry visible indices, prefix tokens need explicit handling, and the
scatter/convolution/gather path needs compile, gradient, checkpoint, and DDP
tests. The dense Canon operation itself covers only 96 sites, so native
PyTorch/cuDNN is the right first implementation. A fused Triton kernel is worth
considering only if profiling shows those small launches are material.

The mask normalization follows partial-convolution practice: use only valid
neighbors, divide by the valid support, and define an all-masked neighborhood
explicitly. Submanifold sparse convolution is the other established choice: it
updates active sites without allowing the active set to expand. For random 75%
audio masking, partial normalization is the closer semantic match.

## Decoder reuse decision

The default is a new lightweight decoder. Reusing an AudioMAE decoder would
only make sense if all of these match:

- encoder output width;
- 12-by-8 patch geometry and patch target dimension;
- prefix-token and positional layout;
- decoder width, depth, head count, and normalization;
- the semantics of the encoder features presented to the decoder.

Available AudioMAE checkpoints do not establish that compatibility for this
Canon-adapted tiny/small/base/large family. Partial loading would give the
appearance of reuse while silently discarding or misaligning most learned
structure.

The trainer does support strict reuse from its own checkpoints. Decoder state
is stored separately, and `--decoder-checkpoint` compares exact architecture
metadata before loading. Full `--resume-from` additionally restores the
encoder, optimizer, scaler, and training step.

## Initialization policy

The default direct run is true random initialization for both encoder and
decoder. Two explicit alternatives are available:

1. `--encoder-pretrained` adapts timm ImageNet weights to the one-channel audio
   input before Canon training.
2. `--encoder-checkpoint` loads a complete architecture-compatible encoder,
   including Canon parameters.

This makes experiments comparable: random direct pretraining is not silently
mixed with image initialization, and decoder reuse never occurs implicitly.
