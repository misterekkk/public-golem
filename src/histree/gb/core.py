import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def transform_bin_mapper(X, thresholds_matrix, counts_vector):

    n_samples, n_features = X.shape

    X_binned = np.zeros((n_samples, n_features), dtype=np.uint8)

    for i in prange(n_samples):

        for j in range(n_features):

            value = X[i, j]
            count = counts_vector[j]

            low = 0
            high = count

            while low < high:
                mid = (low + high) // 2

                if thresholds_matrix[j, mid] <= value:
                    low = mid + 1

                else:
                    high = mid

            X_binned[i, j] = low

    return X_binned


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def build_histograms(
    X_binned, gradients, hessians, indices, features_indices, n_bins=256
):

    n_features = len(features_indices)

    hist = np.zeros((n_features, n_bins, 3))

    for i in prange(n_features):

        feature_idx = features_indices[i]

        for idx in indices:

            bin_idx = X_binned[idx, feature_idx]

            hist[i, bin_idx, 0] += gradients[idx]
            hist[i, bin_idx, 1] += hessians[idx]
            hist[i, bin_idx, 2] += 1

    return hist


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def find_best_split(
    hist, features_indices, reg_lambda, min_samples_leaf, min_gain_to_split
):

    n_features, n_bins, _ = hist.shape

    local_gains = np.full(n_features, -np.inf)
    local_bins = np.zeros(n_features, dtype=np.int32)

    for i in prange(n_features):

        g_parent = np.sum(hist[i, :, 0])
        h_parent = np.sum(hist[i, :, 1])
        count_parent = np.sum(hist[i, :, 2])

        g_left = hist[i, 0, 0]
        h_left = hist[i, 0, 1]
        count_left = hist[i, 0, 2]

        best_gain_i = -np.inf
        best_bin_i = 0

        for j in range(n_bins - 1):

            g_left += hist[i, j, 0]
            h_left += hist[i, j, 1]
            count_left += hist[i, j, 2]
            count_right = count_parent - count_left

            if count_left < min_samples_leaf or count_right < min_samples_leaf:
                continue

            g_right = g_parent - g_left
            h_right = h_parent - h_left

            gain = (
                (g_left**2) / (h_left + reg_lambda)
                + (g_right**2) / (h_right + reg_lambda)
                - (g_parent**2) / (h_parent + reg_lambda)
            )

            if gain > best_gain_i:

                best_gain_i = gain
                best_bin_i = j

        local_gains[i] = best_gain_i
        local_bins[i] = best_bin_i

    best_idx = np.argmax(local_gains)
    best_feature = features_indices[best_idx]
    best_gain = local_gains[best_idx]
    best_split = (best_feature, local_bins[best_idx])

    if best_gain < min_gain_to_split:

        return (-1, -1), -np.inf

    return best_split, best_gain


@njit(fastmath=True, cache=True, nogil=True)
def split_indices(X_binned, indices, start_idx, end_idx, best_feature, best_bin):

    left_count = 0
    for i in range(start_idx, end_idx):

        idx = indices[i]

        if X_binned[idx, best_feature] <= best_bin:

            swap_pos = start_idx + left_count

            temp = indices[swap_pos]
            indices[swap_pos] = indices[i]
            indices[i] = temp

            left_count += 1

    split_idx = start_idx + left_count

    return split_idx


@njit(fastmath=True, cache=True, nogil=True)
def compute_leaf_weight(indices, gradients, hessians, reg_lambda):

    weight = -np.sum(gradients[indices]) / (np.sum(hessians[indices]) + reg_lambda)

    return weight


@njit(cache=True, nogil=True)
def predict_sample(x, tree_structure):

    node_idx = 0

    while True:

        feature_idx = int(tree_structure[node_idx, 0])

        if feature_idx == -1:

            return tree_structure[node_idx, 4]

        threshold = tree_structure[node_idx, 1]

        if x[feature_idx] <= threshold:

            node_idx = int(tree_structure[node_idx, 2])

        else:

            node_idx = int(tree_structure[node_idx, 3])


@njit(parallel=True, cache=True, nogil=True)
def predict_tree(X_binned, tree_structure):

    n_samples = X_binned.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        predictions[i] = predict_sample(X_binned[i], tree_structure)

    return predictions


@njit(fastmath=True, cache=True, nogil=True)
def should_split_pre(
    indices,
    current_depth,
    max_depth,
    min_samples_split,
):

    if max_depth == -1:
        return min_samples_split <= indices.size

    return current_depth < max_depth and min_samples_split <= indices.size


@njit(fastmath=True, cache=True, nogil=True)
def should_split_post(
    indices, best_gain, current_depth, max_depth, min_samples_split, min_gain_split
):

    if max_depth == -1:
        return min_gain_split < best_gain and min_samples_split <= indices.size

    return (
        current_depth < max_depth
        and min_gain_split < best_gain
        and min_samples_split <= indices.size
    )


@njit(nogil=True, cache=True, fastmath=True)
def build_tree(
    X_binned,
    gradients,
    hessians,
    indices,
    features_indices,
    max_depth,
    n_bins,
    min_samples_split,
    min_samples_leaf,
    min_gain_to_split,
    reg_lambda,
):

    if max_depth != -1:
        max_nodes = 2 ** (max_depth + 1)
    else:
        max_nodes = 2 * len(gradients) + 1

    max_nodes = min(max_nodes, 2 * len(gradients) + 1)

    tree = np.zeros((max_nodes, 5), dtype=np.float32)
    tree[:] = -1
    node_count = 1

    root_indices = indices[:]

    root_hist = build_histograms(
        X_binned, gradients, hessians, root_indices, features_indices, n_bins
    )

    # [0] feature_idx    - indeks cechy do podziału (-1 dla liścia)
    # [1] threshold      - próg binu (-1 dla liścia)
    # [2] left_child     - indeks lewego dziecka (-1 dla liścia)
    # [3] right_child    - indeks prawego dziecka (-1 dla liścia)
    # [4] value          - wartość liścia (0.0 dla węzła wewnętrznego)

    # (node_id, depth, start_idx, end_idx, hist)
    stack = [(0, 0, 0, len(indices), root_hist)]

    while len(stack) > 0:

        node_id, depth, start_idx, end_idx, current_hist = stack.pop()

        node_indices = indices[start_idx:end_idx]

        if not should_split_pre(
            indices[start_idx:end_idx],
            current_depth=depth,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        ):

            weight = compute_leaf_weight(node_indices, gradients, hessians, reg_lambda)

            tree[node_id] = [-1, -1, -1, -1, weight]

            continue

        best_split, best_gain = find_best_split(
            current_hist,
            features_indices,
            reg_lambda,
            min_samples_leaf,
            min_gain_to_split,
        )

        best_feature, best_bin = best_split

        if not should_split_post(
            node_indices,
            best_gain=best_gain,
            current_depth=depth,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_gain_split=min_gain_to_split,
        ):

            weight = compute_leaf_weight(node_indices, gradients, hessians, reg_lambda)

            tree[node_id] = [-1, -1, -1, -1, weight]

            continue

        if node_count + 2 >= max_nodes:

            weight = compute_leaf_weight(node_indices, gradients, hessians, reg_lambda)

            tree[node_id] = [-1, -1, -1, -1, weight]

            continue

        left_child_idx = node_count
        right_child_idx = node_count + 1
        node_count += 2

        tree[node_id] = [best_feature, best_bin, left_child_idx, right_child_idx, 0.0]

        split_idx = split_indices(
            X_binned, indices, start_idx, end_idx, best_feature, best_bin
        )

        n_left = split_idx - start_idx
        n_right = end_idx - split_idx

        if n_left < n_right:
            hist_left = build_histograms(
                X_binned,
                gradients,
                hessians,
                indices[start_idx:split_idx],
                features_indices,
                n_bins,
            )

            hist_right = current_hist - hist_left

        else:
            hist_right = build_histograms(
                X_binned,
                gradients,
                hessians,
                indices[split_idx:end_idx],
                features_indices,
                n_bins,
            )

            hist_left = current_hist - hist_right

        if split_idx < end_idx:
            stack.append((right_child_idx, depth + 1, split_idx, end_idx, hist_right))
        else:
            pass

        if start_idx < split_idx:
            stack.append((left_child_idx, depth + 1, start_idx, split_idx, hist_left))
        else:
            pass

    return tree[:node_count]


@njit(nogil=True, cache=True, fastmath=True, parallel=True)
def predict_forest(X_binned, forest_structure):

    n_samples = X_binned.shape[0]
    n_trees = forest_structure.shape[0]

    predictions = np.zeros(n_samples, dtype=np.float32)

    for i in prange(n_samples):

        prediction_sum = 0.0

        for t in range(n_trees):

            node_idx = 0

            while True:

                if forest_structure[t, node_idx, 0] == -1:
                    prediction_sum += forest_structure[t, node_idx, 4]
                    break

                feature_idx = int(forest_structure[t, node_idx, 0])
                threshold = forest_structure[t, node_idx, 1]

                if X_binned[i, feature_idx] <= threshold:
                    node_idx = int(forest_structure[t, node_idx, 2])

                else:
                    node_idx = int(forest_structure[t, node_idx, 3])

        predictions[i] = prediction_sum

    return predictions
