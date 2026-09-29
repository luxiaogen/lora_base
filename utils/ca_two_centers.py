"""Two deterministic class subcenters with a shared residual covariance."""
import torch


@torch.no_grad()
def fit_two_centers(vectors):
    x = torch.as_tensor(vectors, dtype=torch.float64, device='cpu')
    mean = x.mean(0)
    first = (x - mean).square().sum(1).argmax()
    second = (x - x[first]).square().sum(1).argmax()
    centers = torch.stack((x[first], x[second]))
    # Deterministic farthest-point initialization; no training RNG is consumed.
    for _ in range(20):
        labels = (x[:, None] - centers).square().sum(2).argmin(1)
        updated = centers.clone()
        for group in range(2):
            members = x[labels == group]
            if len(members):
                updated[group] = members.mean(0)
        if torch.equal(updated, centers):
            break
        centers = updated

    # centers are means of the last partition, including when iteration 20 ends.
    counts = torch.bincount(labels, minlength=2)
    probs = counts.double() / len(x)
    offsets = centers - mean
    residuals = x - centers[labels]
    between = (offsets.T * probs) @ offsets
    denominator = max(len(x) - 1, 1)
    # The extra between/(n-1) term matches torch.cov's unbiased denominator:
    # shared + between = original sample covariance (+ the existing jitter).
    shared = (residuals.T @ residuals + between) / denominator
    shared += torch.eye(x.shape[1], dtype=x.dtype) * 1e-3
    return offsets.float(), probs.float(), shared.float()


@torch.no_grad()
def add_center_offsets(samples, offsets, probs, generator):
    groups = torch.multinomial(probs.cpu(), len(samples), replacement=True, generator=generator)
    return samples + offsets[groups].to(device=samples.device, dtype=samples.dtype)
