def get_model(model_name, args):
    name = model_name.lower()
    # if name == 'macil':
    #     from methods.macil import Learner
    if name == 'dlora':
        from methods.dlora import Learner
    # elif name == 'lora_init_accum':
    #     from ideas.lora_init_accum.learner import Learner
    elif name == 'dual_mask_branch':
        from ideas.dual_mask_branch.learner import Learner
    return Learner(args)

