"""Current-frame geometric proxies; no ball flight or pass-success interpretation."""

from __future__ import annotations

import numpy as np

THRESHOLDS = np.array([2., 3., 5.], np.float32)
ENDPOINT_EXCLUSION_M = 2.


def lane_distances(actor, candidate, defenders):
    """Finite segment distances with a two-meter exclusion at each endpoint."""
    p, q, d = np.asarray(actor), np.asarray(candidate), np.asarray(defenders)
    delta = q-p
    squared = np.sum(delta*delta, axis=-1)
    length = np.sqrt(squared)
    u = np.sum((d-p)*delta, axis=-1) / np.maximum(squared, 1e-12)
    projection = p+np.clip(u, 0, 1)[..., None]*delta
    distance = np.linalg.norm(d-projection, axis=-1)
    interior = (length > 2*ENDPOINT_EXCLUSION_M) & (u*length > ENDPOINT_EXCLUSION_M) & ((1-u)*length > ENDPOINT_EXCLUSION_M)
    return distance, interior


def geometric_targets(actor, candidate, defenders, visible, ever_seen, ages):
    distance, interior = lane_distances(actor, candidate, defenders)
    obstruction = interior[:, None] & (distance[:, None] <= THRESHOLDS)
    visible_hit = (obstruction & visible[:, None]).any(0)
    hidden_hit = (obstruction & ~visible[:, None]).any(0)
    source = visible_hit.astype(np.int8)+2*hidden_hit.astype(np.int8)
    pressure = float(np.linalg.norm(defenders-candidate, axis=-1).min()) if len(defenders) else 100.
    primary = obstruction[:, 1]
    responsible_hidden = primary & ~visible
    return {
        "lane": obstruction.any(0).astype(np.float32),
        "hidden": hidden_hit.astype(np.float32), "source": source,
        "pressure": np.float32(pressure),
        "never_seen_threat": bool((responsible_hidden & ~ever_seen).any()),
        "responsible_age": np.float32(np.max(ages[responsible_hidden & ever_seen])) if (responsible_hidden & ever_seen).any() else np.float32(-1),
        "boundary": (interior[:, None] & (np.abs(distance[:, None]-THRESHOLDS) <= 1.)).any(0),
    }


def probability_from_samples(actor, candidate, samples, valid, hidden, occupancy_samples=None):
    """Independent Gaussian entities; optional independent anonymous-cell draws."""
    distance, interior = lane_distances(actor, candidate, samples)
    hits = interior[..., None] & (distance[..., None] <= THRESHOLDS) & valid[None, :, None]
    all_hit, hidden_hit = hits.any(1), (hits & hidden[None, :, None]).any(1)
    pressure = np.where(valid[None], np.linalg.norm(samples-candidate, axis=-1), 100.).min(1)
    if occupancy_samples is not None:
        positions, selected = occupancy_samples
        distance, interior = lane_distances(actor, candidate, positions)
        anonymous_hit = (interior[..., None] & (distance[..., None] <= THRESHOLDS) & selected[..., None]).any(1)
        all_hit |= anonymous_hit
        hidden_hit |= anonymous_hit
        anonymous_pressure = np.where(selected, np.linalg.norm(positions-candidate, axis=-1), 100.).min(1)
        pressure = np.minimum(pressure, anonymous_pressure)
    return np.r_[all_hit.mean(0), hidden_hit.mean(0), pressure.mean()].astype(np.float32)
