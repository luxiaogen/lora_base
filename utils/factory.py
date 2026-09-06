def get_model(model_name, args):
    name = model_name.lower()
    if name in ('dlora', 'dual_mask_branch'):
        from methods.dlora import Learner
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    return Learner(args)
