import os
import os.path
import sys
import logging
import copy
import json
import time
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import random

def _init_experiment_tracker(args):
    backend = str(args.get('experiment_tracker', 'none')).strip().lower()
    if backend not in ('none', 'tensorboard', 'wandb'):
        raise ValueError('experiment_tracker must be one of: none, tensorboard, wandb')
    if backend == 'none':
        return None
    # 实验名称
    experiment_name = str(args.get('prefix') or args['model_name'])
    if backend == 'tensorboard':
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                'experiment_tracker=tensorboard requires tensorboard. '
                'Run: pip install tensorboard'
            ) from exc

        log_dir = os.path.join(
            str(args.get('tensorboard_logdir', 'logs/tensorboard')),
            str(args['dataset']),
            '{}_tasks'.format(args['total_sessions']),
            experiment_name,
            'seed_{}'.format(args['seed']),
        )
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text(
            'config',
            '```json\n{}\n```'.format(
                json.dumps(args, default=str, indent=2, sort_keys=True)
            ),
            global_step=0,
        )
        return backend, writer

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError('experiment_tracker=wandb requires wandb. Run: pip install wandb') from exc

    group = args.get('wandb_group') or '{}_{}tasks_{}'.format(args['dataset'], args['total_sessions'], experiment_name)
    tags = args.get('wandb_tags', [args['dataset'], args['model_name']])
    if isinstance(tags, str):
        tags = [item.strip() for item in tags.split(',') if item.strip()]
    run = wandb.init(
        project=str(args.get('wandb_project', 'LoDA_ICML2026')),
        entity=args.get('wandb_entity') or None,
        name='{}_{}_seed{}'.format(args['dataset'], experiment_name, args['seed']),
        group=str(group),
        tags=list(tags),
        config=copy.deepcopy(args),
        mode=str(args.get('wandb_mode', 'online')),
    )
    return backend, run


def _write_experiment_metrics(tracker, metrics, step):
    backend, handle = tracker
    if backend == 'wandb':
        handle.log(metrics, step=int(step))
        return

    for name, value in metrics.items():
        handle.add_scalar(name, value, global_step=int(step))
    handle.flush()


def _close_experiment_tracker(tracker):
    if tracker is None:
        return
    backend, handle = tracker
    if backend == 'wandb':
        handle.finish()
    else:
        handle.close()


def _last_metric(model, attribute):
    values = getattr(model, attribute, None)
    if values is None or len(values) == 0:
        return None
    return float(values[-1])


def _log_experiment_task(
    tracker,
    model,
    task_id,
    cnn_accy,
    cnn_accy_with_task,
    task_prediction_accuracy,
    w0_accuracy,
    train_seconds,
    eval_seconds,
    forgetting=None,
    backward=None,
):
    if tracker is None:
        return

    grouped = cnn_accy.get('grouped', {})
    metrics = {
        'task/index': int(task_id),
        'accuracy/top1': float(cnn_accy['top1']),
        'accuracy/with_task_top1': float(cnn_accy_with_task['top1']),
        'task_prediction/accuracy': float(task_prediction_accuracy) * 100.0,
        'time/train_seconds': float(train_seconds),
        'time/eval_seconds': float(eval_seconds),
    }
    for name in ('old', 'new'):
        if name in grouped:
            metrics['accuracy/{}'.format(name)] = float(grouped[name])
    if w0_accuracy is not None:
        metrics['w0/ncm_top1'] = float(w0_accuracy)
    if forgetting is not None:
        metrics['continual/forgetting'] = float(forgetting)
        metrics['continual/backward'] = float(backward)

    competence = getattr(model, '_w0_competence', None)
    if competence is not None:
        metrics['w0/competence'] = float(competence) * 100.0


    for attribute, name in (
        ('_w0_competence_new', 'w0/competence_new'),
        ('_w0_competence_all_seen', 'w0/competence_all_seen'),
        ('_w0_old_overlap_risk', 'w0/old_overlap_risk'),
        ('_w0_control_competence', 'w0/control_competence'),
    ):
        value = getattr(model, attribute, None)
        if value is not None:
            metrics[name] = float(value) * 100.0
    for attribute, name in (
        ('_w0_ncm_loss_new', 'w0/ncm_loss_new'),
        ('_w0_plasticity_demand', 'w0/plasticity_demand'),
    ):
        value = getattr(model, attribute, None)
        if value is not None:
            metrics[name] = float(value)

    functional_merge = getattr(model, '_functional_merge_calibration', None)
    if functional_merge is not None:
        for name, value in functional_merge.items():
            metrics['dual_mask/functional_merge/{}'.format(name)] = float(value)

    for attribute, name in (
        ('_feature_drift_curve', 'w0/feature_drift'),
        ('_weight_drift_curve', 'w0/weight_drift_mean'),
    ):
        value = _last_metric(model, attribute)
        if value is not None:
            metrics[name] = value

    for name, value in getattr(model,'_last_epoch_training_loss_metrics',{},).items():
        metrics['train/{}'.format(name)] = float(value)

    iter_modules = getattr(model, '_iter_lora_modules', None)
    if callable(iter_modules):
        modules = list(iter_modules())
        if modules:
            first_module = modules[0]
            metrics['dual_mask/coverage'] = float(first_module.effective_energy_coverage)
            metrics['dual_mask/protect_strength'] = float(first_module.effective_protect_strength)
            metrics['dual_mask/private_rank'] = int(first_module.current_private_rank)
            for layer_idx, module in enumerate(modules):
                metrics['dual_mask/layer_{}/protect_density'.format(layer_idx)] = float(module.general_mask.detach().float().mean().item())
                metrics['dual_mask/layer_{}/protected_importance_mean'.format(layer_idx)] = float((module.general_mask.detach().float() * module.w0_importance.detach().float()).mean().item())

                for attribute, name in (
                        ('last_conflict_entropy', 'conflict_entropy'),
                        ('last_conflict_top10_energy', 'conflict_top10_energy'),
                        ('last_conflict_energy50_ratio', 'conflict_energy50_ratio'),
                        ('last_conflict_gate_suppression', 'conflict_gate_suppression'),
                        ('last_safe_suppression', 'safe_suppression'),
                        ('last_effective_conflict_ratio', 'effective_conflict_ratio'),
                        ('last_effective_conflict_strength', 'effective_conflict_strength'),
                        ('last_private_conflict_mask_overlap', 'private_conflict_mask_overlap'),
                    ('last_private_conflict_energy_overlap', 'private_conflict_energy_overlap'),
                    ('last_private_conflict_gate_suppression', 'private_conflict_gate_suppression'),
                ):
                    value = getattr(module, attribute, None)
                    if value is not None:
                        metrics['dual_mask/layer_{}/{}'.format(layer_idx, name)] = (float(value.detach().item()))

    _write_experiment_metrics(tracker, metrics, task_id)


def _log_experiment_summary(
    tracker,
    cnn_curve,
    w0_curve,
    task_prediction_curve,
    elapsed_seconds,
):
    if tracker is None:
        return

    metrics = {
        'summary/average_accuracy': float(np.mean(cnn_curve)),
        'summary/last_accuracy': float(cnn_curve[-1]),
        'summary/task_prediction_average': float(np.mean(task_prediction_curve)) * 100.0,
        'summary/task_prediction_last': float(task_prediction_curve[-1]) * 100.0,
        'time/run_seconds': float(elapsed_seconds),
    }
    if w0_curve:
        metrics['summary/w0_ncm_average'] = float(np.mean(w0_curve))
        metrics['summary/w0_ncm_last'] = float(w0_curve[-1])
    _write_experiment_metrics(tracker, metrics, len(cnn_curve))


def train(args):
    start_time = time.time()
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        experiment_tracker = _init_experiment_tracker(args)
        try:
            _train(args, experiment_tracker=experiment_tracker)
        finally:
            _close_experiment_tracker(experiment_tracker)

    total_time = time.time() - start_time
    logging.info('Total experiment time: {:.2f}s ({:.2f}min, {:.2f}h)'.format(total_time, total_time / 60, total_time / 3600))

def _train(args, experiment_tracker=None):
    run_start_time = time.time()

    if args["ca"]:
        prefix_head = "ca"
    else:
        prefix_head = "standard_head"

    prefix_lora = args["prefix"] # rank=64
    if prefix_lora != "":
        prefix_lora += "_"

    if not args["use_slora"] and not args["use_plora"]:
        prefix_lora += 'seq_lora'
    else:
        pass

    # rank=64_seq_lora
    logdir = 'logs/{}/{}_tasks/{}/{}'.format(args['dataset'], args['total_sessions'], prefix_head, prefix_lora)
    args['logdir'] = logdir # logs/ImageNet_R/20_tasks/standard_head/rank=64_seq_lora
    print(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}_slora:{}_plora:{}_rank:{}_{}_{}_{}-{}'.format(args['seed'], args["use_slora"], args["use_plora"], args['rank'], args.get("lora_type", "lora"), args['model_name'], args['optim'], args['lrate']))
    #logfilename = os.path.join(logdir, '{}_slora:{}_plora:{}_rank:{}_{}_{}_{}-{}'.format(args['seed'], args["use_slora"], args["use_plora"], args['rank'], args["lora_type"], args['model_name'], args['optim'], args['lrate']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        force=True,
        handlers=[ # （双路输出处理器,把日志同时分发给两个目标
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    ) # logs/CUB/10_tasks/ca/idea3_wpre_adaptive_/1993_slora:True_plora:True_rank:32_lora_dual_mask_branch_sgd-0.005
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args['dataset'],
        args['shuffle'],
        args['seed'],
        args['init_cls'],
        args['increment'],
        args
    )
    model = factory.get_model(args['model_name'], args)


    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    w0_curve = []
    for task_id in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        train_seconds = time_end - time_start
        logging.info('Time:{}'.format(train_seconds))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()

        w0_accuracy = None
        if bool(args.get('dual_mask_track_w0_metrics', False)):
            w0_accuracy = model.eval_w0_task()
            if w0_accuracy is not None:
                w0_curve.append(round(w0_accuracy, 2))
                logging.info('W_pre-only NCM top1 curve: {}'.format(w0_curve))

        time_end = time.time()
        eval_seconds = time_end - time_start
        logging.info('Time:{}'.format(eval_seconds))
        # raise Exception
        model.after_task()

        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
        cnn_curve_task['top1'].append(cnn_accy_task)
        # 不知道 task id，在所有已见类里选
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        # 知道真实 task id，只在该 task 的类里选
        logging.info('CNN top1 with task curve: {}'.format(cnn_curve_with_task['top1'])) # 已知 Task-ID 时的局部准确率
        # logging.info('CNN top1 task curve: {}'.format(cnn_curve_task['top1']))
        task_pred_curve = [round(acc * 100, 2) for acc in cnn_curve_task['top1']]
        task_pred_avg = round(float(np.mean(cnn_curve_task['top1']) * 100.0), 2)
        logging.info('Task Prediction Accuracy curve (%): {}'.format(task_pred_curve))

        """
            Forgetting 0.0235 →平均而言，历史旧 Task 的准确率相比各自的历史最高点只下降了 2.35%（抗遗忘性能优秀）
            Backward -0.0210 → 平均而言，当前旧 Task 的准确率比刚学完时低了 2.10%。
        """
        forgetting = None
        backward = None
        if task_id > 0:
            diagonal = np.diag(model.acc_matrix)
            # Forgetting: 历史旧 Task 的历史最高准确率 - 当前测试准确率 的平均值
            forgetting = np.mean((np.max(model.acc_matrix, axis=1) -
                                model.acc_matrix[:, task_id])[:task_id])
            # BWT / 后向知识迁移: 当前测试准确率 - 刚学完该 Task 时的对角线准确率的平均值
            backward = np.mean((model.acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str = "Forgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
            logging.info(result_str)

        _log_experiment_task(
            experiment_tracker,
            model,
            task_id,
            cnn_accy,
            cnn_accy_with_task,
            cnn_accy_task,
            w0_accuracy,
            train_seconds,
            eval_seconds,
            forgetting,
            backward,
        )

    logging.info('Accuracy Matrix: \n {}'.format(model.acc_matrix.T.round(2)))
    logging.info('Average Accuracy: {}'.format(np.mean(cnn_curve['top1'])))
    logging.info('Last Accuracy: {}'.format(cnn_curve['top1'][-1]))

    if w0_curve:
        logging.info('W_pre-only NCM Average Accuracy: {}'.format(np.mean(w0_curve)))
        logging.info('W_pre-only NCM Last Accuracy: {}'.format(w0_curve[-1]))

    logging.info('Task Prediction Accuracy average (%): {:.2f}'.format(task_pred_avg))
    logging.info(
        'Final Task Prediction Accuracy (%): {:.2f}'.format(
            cnn_curve_task['top1'][-1] * 100.0
        )
    )
    _log_experiment_summary(
        experiment_tracker,
        cnn_curve['top1'],
        w0_curve,
        cnn_curve_task['top1'],
        time.time() - run_start_time,
    )

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if str(device) == '-1':
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.get('disable_fused_sdpa', False):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
