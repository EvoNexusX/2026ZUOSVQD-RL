import numpy as np


class _NBCResult:
    def __init__(self, labels, distance_threshold):
        self.labels_ = labels
        self.distance_threshold_ = distance_threshold
        unique = np.unique(labels)
        valid = [int(l) for l in unique if int(l) != -1]
        self.n_clusters = len(valid)
        self.cluster_sizes_ = {int(l): int(np.sum(labels == l)) for l in valid}


def _empty_result():
    return _NBCResult(np.array([], dtype=int), float('inf'))


def nearest_better_clustering(rewards=None, X=None, distance_threshold=None,
                             distance_percentile=90):
    if rewards is None:
        return _empty_result()
    rewards = np.asarray(rewards).astype(float)
    n = rewards.shape[0]
    if n == 0:
        return _empty_result()

    if X is None:
        X = rewards.reshape(-1, 1)
    else:
        X = np.asarray(X).astype(float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    diffs = X[:, None, :] - X[None, :, :]
    D = np.linalg.norm(diffs, axis=2)

    better_neighbors = -np.ones(n, dtype=int)
    better_dists = np.zeros(n, dtype=float)
    for i in range(n):
        cand = np.where(rewards > rewards[i])[0]
        if cand.size == 0:
            continue
        j = cand[np.argmin(D[i, cand])]
        better_neighbors[i] = int(j)
        better_dists[i] = float(D[i, j])

    finite_d = better_dists[better_neighbors != -1]
    if distance_threshold is not None:
        thr = float(distance_threshold)
    elif finite_d.size > 0:
        thr = float(np.percentile(finite_d, distance_percentile))
    else:
        thr = float('inf')

    labels = -np.ones(n, dtype=int)
    next_cluster_id = 0
    rev_graph = [[] for _ in range(n)]
    for i in range(n):
        j = better_neighbors[i]
        if j != -1 and better_dists[i] <= thr:
            rev_graph[j].append(i)

    roots = np.where((better_neighbors == -1) | (better_dists > thr))[0]
    for r in roots:
        stack = [int(r)]
        while stack:
            node = stack.pop()
            if labels[node] != -1:
                continue
            labels[node] = next_cluster_id
            for ch in rev_graph[node]:
                if better_dists[ch] <= thr:
                    stack.append(ch)
        next_cluster_id += 1

    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_cluster_id
            next_cluster_id += 1

    return _NBCResult(labels, float(thr))
