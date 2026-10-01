"""Small label-free selectors; fit only on explicitly supplied current-train calibration rows."""
import numpy as np


CORE = ('proposal_advantage', 'margin_advantage', 'base_uncertainty')
EXTRA = ('prototype_advantage', 'entropy_advantage')


def signal_view(rows):
    # Inference receives neither ground truth, true-task predictions nor sample indices.
    keys = ('base_prediction', 'anchor_prediction') + CORE + EXTRA
    return {key: np.asarray(rows[key]) for key in keys if key in rows}


def candidate_specs(extended=True):
    specs = []
    for signal in CORE[:2]:
        for threshold in (0., .02, .05, .1):
            specs.append(dict(name=f'{signal}_fixed_{threshold:g}', kind='fixed',
                              signal=signal, threshold=threshold))
        for quantile in (.90, .95, .98, .99):
            specs.append(dict(name=f'{signal}_tail_{quantile:g}', kind='quantile',
                              signal=signal, quantile=quantile))
    for threshold in (0., .02):
        for gap in (.02, .05):
            specs.append(dict(name=f'joint_proposal_{threshold:g}_gap_{gap:g}', kind='joint',
                              signal='proposal_advantage', threshold=threshold, gap=gap))
    for signal in CORE:
        for cost in (1, 3, 10):
            specs.append(dict(name=f'{signal}_utility_cost{cost}', kind='utility',
                              signal=signal, cost=cost))
    for features in (('core', 'extended') if extended else ('core',)):
        for cost in (1, 3, 10):
            specs.append(dict(name=f'ridge_{features}_cost{cost}', kind='ridge',
                              features=features, cost=cost))
    if extended:
        for signal in EXTRA:
            for cost in (1, 3, 10):
                specs.append(dict(name=f'{signal}_utility_cost{cost}', kind='utility',
                                  signal=signal, cost=cost))
    return specs


def outcomes(rows):
    base = rows['base_prediction'] == rows['target']
    anchor = rows['anchor_prediction'] == rows['target']
    return ~base & anchor, base & ~anchor


def fit_policy(spec, calibration):
    policy = dict(spec)
    view = signal_view(calibration)
    disagree = view['base_prediction'] != view['anchor_prediction']
    if spec['kind'] in ('fixed', 'joint'):
        return policy
    policy['threshold'] = None
    if not disagree.any():
        return policy
    if spec['kind'] == 'quantile':
        policy['threshold'] = max(0., float(np.quantile(view[spec['signal']][disagree], spec['quantile'])))
        return policy
    rescued, harmed = outcomes(calibration)
    utility = rescued.astype(float) - spec['cost'] * harmed
    if not rescued[disagree].any():
        return policy
    if spec['kind'] == 'utility':
        values = view[spec['signal']][disagree]
        order = np.argsort(-values, kind='stable')
        scores = values[order]
        gains = utility[disagree][order].cumsum()
        ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
        best = ends[np.argmax(gains[ends])]
        if gains[best] > 0:
            policy['threshold'] = float(np.nextafter(scores[best], -np.inf))
    elif spec['kind'] == 'ridge':
        keys = CORE + EXTRA if spec['features'] == 'extended' else CORE
        features = np.column_stack([view[key][disagree] for key in keys])
        mean, scale = features.mean(0), np.maximum(features.std(0), 1e-6)
        design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ utility[disagree])
        policy.update(threshold=0., keys=list(keys), mean=mean.tolist(), scale=scale.tolist(),
                      coef=coef.tolist())
    return policy


def choose(policy, view):
    disagree = view['base_prediction'] != view['anchor_prediction']
    if policy.get('threshold') is None:
        return np.zeros_like(disagree)
    if policy['kind'] == 'ridge':
        features = np.column_stack([view[key] for key in policy['keys']])
        design = np.column_stack([np.ones(len(features)),
                                  (features - policy['mean']) / policy['scale']])
        selected = design @ np.asarray(policy['coef']) > 0
    else:
        selected = view[policy['signal']] > policy['threshold']
        if policy['kind'] == 'joint':
            selected &= -view['base_uncertainty'] < policy['gap']
    return disagree & selected
