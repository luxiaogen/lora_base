"""Fixed protection-position control and read-only measurements of delivered updates."""
import torch


def permute_protect_mask(mask, seed, layer):
    parts = []
    for projection, source in enumerate(mask.chunk(3, dim=0)):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed) + 104729 * (int(layer) + 1) + 1009 * projection)
        rows = torch.randperm(source.shape[0], generator=generator).to(mask.device)
        columns = torch.randperm(source.shape[1], generator=generator).to(mask.device)
        parts.append(source.index_select(0, rows).index_select(1, columns))
    return torch.cat(parts, dim=0)


def update_rows(raw, base, safe, conflict_mask, protect_mask, **identity):
    rows = []
    parts = zip(raw.chunk(3), base.chunk(3), safe.chunk(3), conflict_mask.chunk(3), protect_mask.chunk(3))
    for projection, (r, b, s, c, p) in zip(('Q', 'K', 'V'), parts):
        raw_norm, base_norm, safe_norm = (float(x.float().norm()) for x in (r, b, s))
        allowed = (1 - p).bool() if identity['branch'] == 'P' else torch.ones_like(p, dtype=torch.bool)
        selected = c.bool() & allowed
        total_removed = float((r - s).float().norm())
        conflict_removed = float((b - s).float().norm())
        rows.append(dict(identity, projection=projection,
            raw_norm=raw_norm, pre_conflict_norm=base_norm, effective_norm=safe_norm,
            total_removed_norm=total_removed, conflict_removed_norm=conflict_removed,
            total_removed_ratio=total_removed / max(raw_norm, 1e-12),
            conflict_removed_ratio=conflict_removed / max(base_norm, 1e-12),
            protect_density=float(p.float().mean()), plastic_density=float((1 - p).float().mean()),
            selected_coordinates=int(c.sum()), effective_selected_coordinates=int(selected.sum()),
            conflict_density=float(c.float().mean()),
            effective_conflict_density=float(selected.float().mean())))
    return rows
