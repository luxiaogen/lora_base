"""Offline feature geometry; never called by training."""
import numpy as np
import torch


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def remap_labels(original_labels, saved_order):
    inverse = {original_id: i for i, original_id in enumerate(saved_order)}
    return np.asarray([inverse[int(c)] for c in original_labels])


@torch.inference_mode()
def extract_features(network, loader, device, split):
    features, targets, sample_ids = [], [], []
    for step, (indices, inputs, labels) in enumerate(loader):
        features.append(network.extract_vector(inputs.to(device), task_id=network.numtask - 1).cpu().float().numpy())
        targets.append(labels.numpy())
        sample_ids.append(indices.numpy())
        if step % 25 == 0 or step + 1 == len(loader):
            print(f'{split}: batch {step + 1}/{len(loader)}', flush=True)
    return np.concatenate(features), np.concatenate(targets), np.concatenate(sample_ids)


def analyze_geometry(train_features, train_labels, test_features, test_labels, heads):
    train, test, weights = map(normalize, (train_features, test_features, heads))
    y = np.asarray(test_labels, dtype=int)
    n_classes = len(weights)
    counts = np.bincount(train_labels, minlength=n_classes)
    centers = np.full((n_classes, train.shape[1]), np.nan)
    dispersion = np.full(n_classes, np.nan)
    for c in range(n_classes):
        points = train[train_labels == c]
        if len(points):
            centers[c] = normalize(points.mean(axis=0, keepdims=True))[0]
            dispersion[c] = np.sqrt(np.mean(np.sum((points - centers[c]) ** 2, axis=1)))
    logits = test @ weights.T
    prediction = logits.argmax(axis=1)
    wrong_logits = logits.copy()
    wrong_logits[np.arange(len(y)), y] = -np.inf
    margin = logits[np.arange(len(y)), y] - wrong_logits.max(axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (y, prediction), 1)
    center_similarity = centers @ centers.T
    other = np.nan_to_num(center_similarity, nan=-np.inf)
    np.fill_diagonal(other, -np.inf)
    nearest = other.argmax(axis=1)
    nearest_distance = np.sqrt(np.maximum(0, 2 - 2 * other.max(axis=1)))
    nearest_distance[counts == 0] = np.nan
    alignment = centers @ weights.T
    competing = alignment.copy()
    np.fill_diagonal(competing, -np.inf)
    center_logits = np.nan_to_num(test @ centers.T, nan=-np.inf)
    return dict(accuracy=float(100 * np.mean(prediction == y)), prediction=prediction,
                margin=margin, confusion=confusion, centers=centers, train_counts=counts,
                dispersion=dispersion, nearest_center=nearest,
                nearest_center_distance=nearest_distance, center_similarity=center_similarity,
                own_head_cosine=alignment.diagonal(), competitor_head=competing.argmax(axis=1),
                competitor_head_cosine=competing.max(axis=1),
                nearest_center_prediction=center_logits.argmax(axis=1))


def top_confused_pairs(confusion, count=3):
    pairs = [(i, j) for i in range(len(confusion)) for j in range(i + 1, len(confusion))
             if confusion[i, j] + confusion[j, i] > 0]
    return sorted(pairs, key=lambda p: (-(confusion[p] + confusion[p[::-1]]), *p))[:count]
