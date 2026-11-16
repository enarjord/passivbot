"""CPU-side bootstrap sampling for wide GPU optimizer cohorts."""

from collections import deque

from optimization.bounds import enforce_bounds


def _sample_gene(bound, random_state, *, old_value=None):
    if not bound.is_stepped:
        return float(random_state.uniform(bound.low, bound.high))
    if old_value is None:
        index = int(random_state.integers(bound.max_index + 1))
    else:
        old_index = round(bound.value_to_index(old_value))
        # Draw from the grid with the parent's valid index removed.
        index = int(random_state.integers(bound.max_index))
        if index >= old_index:
            index += 1
    return bound.index_to_value(index)


def max_starting_vectors(starts, *, bounds, sig_digits, width, random_state, preserve_indices=()):
    """Preserve supplied starts and append distinct, repaired bootstrap candidates.

    Supplied vectors have already been resolved and repaired by the seed loader.
    With seeds, every appended vector changes one gene of a seed or its descendant.
    ``preserve_indices`` can keep authored anchor associations on that path.
    Without seeds, sample the configured bounds using the caller's random generator.
    Narrow or quantized search spaces may exhaust distinct proposals before width.
    """
    if type(width) is not int or width < 1:
        raise ValueError("GPU bootstrap width must be a positive integer")
    vectors = list(starts)
    if len(vectors) >= width:
        return vectors
    low = enforce_bounds([bound.low for bound in bounds], bounds, sig_digits)
    high = enforce_bounds([bound.high for bound in bounds], bounds, sig_digits)
    seeded = bool(vectors)
    frozen = set(preserve_indices) if seeded else set()
    mutable = [i for i, (lo, hi) in enumerate(zip(low, high, strict=True)) if lo != hi and i not in frozen]
    if not mutable:
        return vectors or [low]

    # In a sparse Boolean lattice, uniformly drawing a parent from every
    # accepted vector increasingly reselects exhausted interiors. Retain a
    # bounded rotating list of recently accepted parents instead.
    parents = deque(vectors, maxlen=max(128, len(vectors))) if seeded else None
    seen = {tuple(vector) for vector in vectors}
    misses = 0
    # Bound the effort even for heavily quantized spaces. Each success resets
    # the limit so a useful series of distinct proposals can fill the cohort.
    missed_proposal_limit = 128
    while len(vectors) < width and misses < missed_proposal_limit:
        if seeded:
            parent = parents.popleft()
            parents.append(parent)
            candidate = list(parent)
            gene = mutable[int(random_state.integers(len(mutable)))]
            candidate[gene] = _sample_gene(bounds[gene], random_state, old_value=candidate[gene])
        else:
            candidate = low.copy()
            for gene in mutable:
                candidate[gene] = _sample_gene(bounds[gene], random_state)
        candidate = enforce_bounds(candidate, bounds, sig_digits)
        key = tuple(candidate)
        if key in seen:
            misses += 1
            continue
        vectors.append(candidate)
        seen.add(key)
        if seeded:
            parents.append(candidate)
        misses = 0
    return vectors
