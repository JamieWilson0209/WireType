"""Collapse diagnostics for the embedding covariance.

An encoder can map every neuron to the same point, or to a low-dimensional
subspace of its output. Both show up in the eigenspectrum of the covariance of a
batch of embeddings before they show up in a probe, so the training curve
(`scripts/train.py`) reports these at every probe step.

Read the three numbers together:

    effective_rank      exp of the entropy of the normalised spectrum. The
                        headline: how many directions the representation
                        actually uses. Falls towards 1 under full collapse.
    stable_rank         total variance over the largest eigenvalue. Sensitive
                        to one direction running away from the rest.
    smallest_ratio      the fraction of usable directions carrying less than
                        `dead_threshold` of the mean eigenvalue. Dimensional
                        collapse, where most of the spectrum is dead but the
                        surviving directions look healthy.

All three are computed on centred embeddings, so they describe the spread of
the representation, not its offset from the origin.

## Why every rank here is reported against a null

A covariance over `d` channels estimated from `n` neurons has at most
`min(n - 1, d)` non-zero directions, however healthy the representation is.
The missing directions are an artefact of counting, not a property of the
encoder, and they are large: at d = 2,048 and a batch of 1,024, embeddings
drawn from pure isotropic noise — full rank by construction, nothing wrong
with them at all — report an effective rank of 797 and read as 50% dead.

So a raw rank cannot be compared against a fixed fraction of `d`. Every rank
below is therefore also reported as a ratio against **what the same measurement
returns on isotropic noise at the same batch size and width**.
`collapse_ratio` is the headline: it reads ~1.0 for a healthy representation at
any (n, d), and falls towards 0 as the encoder collapses.

That null costs almost nothing to compute, because it depends only on the
aspect ratio n/d and not on n and d separately — verified stable to 0.5% over
a 16-fold range of d — so it is evaluated once at a small reference size and
rescaled.

**The diagnostic needs headroom.** At n = 4d the null sits at 88% of the
maximum attainable rank, leaving most of the range to detect collapse in. At
n < d it sits above 94% and there is very little range left. Estimate these
metrics on a held-out set of at least 4d neurons rather than on the training
batch.
"""

from __future__ import annotations

from functools import lru_cache

import torch

# Reference width for the null. The null depends only on n/d, so this is an
# accuracy/cost knob and nothing else; 256 is within 0.5% of the large-d limit
# and its eigendecomposition is instant.
_NULL_REFERENCE_DIMS = 256
_NULL_DRAWS = 3


@torch.no_grad()
def _centred_spectrum(embeddings: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of the covariance of centred `embeddings`, descending.

    Taken in float64 — the small eigenvalues are the informative end, and in
    float32 they are numerically indistinguishable from zero.
    """
    centred = embeddings.detach().to(torch.float64)
    centred = centred - centred.mean(dim=0, keepdim=True)

    n_nodes = centred.shape[0]
    if n_nodes < 2:
        raise ValueError("collapse metrics need at least 2 neurons, got %d" % n_nodes)

    covariance = centred.T @ centred / (n_nodes - 1)
    try:
        eigenvalues = torch.linalg.eigvalsh(covariance)
    except Exception:
        # LAPACK's symmetric eigensolver can fail to converge on a covariance
        # with repeated or near-repeated eigenvalues, which is exactly what a
        # partly collapsed representation produces — so the diagnostic is most
        # likely to crash on the runs it exists to catch. A real run died here.
        # Singular values of a symmetric positive semi-definite matrix are its
        # eigenvalues, and the SVD path is robust to the degeneracy that breaks
        # the eigensolver, at the same O(d^3). Falling back rather than adding
        # jitter keeps the numbers exact when it is used.
        eigenvalues = torch.linalg.svdvals(covariance)
    return eigenvalues.clamp_min(0.0).flip(0) if eigenvalues[0] <= eigenvalues[-1] else eigenvalues.clamp_min(0.0)


def _rank_stats(eigenvalues: torch.Tensor) -> tuple[float, float]:
    """(effective rank, stable rank) from a spectrum. Both 0.0 on a point mass.

    Under total collapse every embedding is identical, the centred covariance
    is exactly zero, and both quantities are 0/0. They are reported as 0.0
    rather than as NaN, because NaN fails every `<` comparison and would let
    the worst possible outcome slip past the stop rule silently.
    """
    total_variance = eigenvalues.sum()
    if total_variance <= 0.0:
        return 0.0, 0.0

    spectrum = eigenvalues / total_variance
    entropy = -(spectrum * torch.log(spectrum.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy)), float(total_variance / eigenvalues.max())


@lru_cache(maxsize=None)
def null_rank_stats(n_samples: int, n_dims: int) -> tuple[float, float]:
    """What `_rank_stats` returns on isotropic noise at this batch size and width.

    The healthy reference: embeddings with no structure and no collapse, whose
    only rank deficiency is the one that comes from estimating a `d` x `d`
    covariance from `n` neurons. Returns (effective rank, stable rank).

    Evaluated at a small reference width and rescaled, which is exact up to
    Monte-Carlo noise because both quantities are `max_rank` times a function
    of the aspect ratio alone. Cached per (n_samples, n_dims).
    """
    max_rank = min(n_samples - 1, n_dims)
    reference_dims = min(n_dims, _NULL_REFERENCE_DIMS)
    reference_samples = max(2, round(n_samples / n_dims * reference_dims))
    reference_max_rank = min(reference_samples - 1, reference_dims)

    effective, stable = 0.0, 0.0
    for draw in range(_NULL_DRAWS):
        generator = torch.Generator().manual_seed(draw)
        noise = torch.randn(
            reference_samples, reference_dims, generator=generator, dtype=torch.float64
        )
        one_effective, one_stable = _rank_stats(_centred_spectrum(noise))
        effective += one_effective / _NULL_DRAWS
        stable += one_stable / _NULL_DRAWS

    scale = max_rank / reference_max_rank
    return effective * scale, stable * scale


@torch.no_grad()
def collapse_metrics(
    embeddings: torch.Tensor,
    dead_threshold: float = 0.01,
) -> dict[str, float]:
    """Summarise the covariance spectrum of one set of embeddings.

    `embeddings` is (n_nodes, n_dims). Prefer a held-out set of at least
    4 x n_dims neurons over the training batch — see the module docstring.

    `collapse_ratio` is the headline and the one the stop rule reads: the
    effective rank as a fraction of what isotropic noise scores at the same
    batch size and width. ~1.0 is healthy, 0.0 is total collapse.
    """
    eigenvalues = _centred_spectrum(embeddings)
    n_nodes, n_dims = embeddings.shape
    max_rank = min(n_nodes - 1, n_dims)

    effective_rank, stable_rank = _rank_stats(eigenvalues)
    null_effective_rank, null_stable_rank = null_rank_stats(n_nodes, n_dims)

    # Count dead directions only among the ones that could have been alive.
    # Everything past `max_rank` is a structural zero and says nothing about
    # the encoder: including them made this read 88% dead on healthy noise.
    # On a point mass every direction is dead, but the threshold would then be
    # comparing 0 against 0 and reporting the healthiest possible value, so
    # that case is named rather than left to the arithmetic.
    usable = eigenvalues[:max_rank]
    if usable.mean() <= 0.0:
        dead_fraction = 1.0
    else:
        dead_fraction = float(
            (usable < dead_threshold * usable.mean()).to(torch.float64).mean()
        )

    return {
        "effective_rank": effective_rank,
        "collapse_ratio": effective_rank / null_effective_rank,
        "stable_rank": stable_rank,
        "stable_ratio": stable_rank / null_stable_rank,
        "smallest_ratio": dead_fraction,
        "total_variance": float(eigenvalues.sum()),
        "n_dims": int(n_dims),
        "n_samples": int(n_nodes),
        "max_rank": int(max_rank),
        "null_effective_rank": null_effective_rank,
    }
