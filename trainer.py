import os
import os.path
import sys
import logging
import copy
import time
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import time

def train(args):
    start_time = time.time()
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    total_time = time.time() - start_time
    logging.info('Total experiment time: {:.2f}s ({:.2f}min, {:.2f}h)'.format(
        total_time, total_time / 60, total_time / 3600
    ))

def _train(args):

    if args["ca"]:
        prefix_head = "ca"
    else:
        prefix_head = "standard_head"

    prefix_lora = args["prefix"] # rank=64
    if prefix_lora != "":
        prefix_lora += "_"

    # if args.get("use_flat", False) or args.get("use_gao", False):
    #     prefix_lora += ("rho=" + str(args.get("rho", 0)) + "_") # GAO的超参数

    if not args["use_slora"] and not args["use_plora"]:
        prefix_lora += 'seq_lora'
    else:
        pass
        # if args["use_slora"] and not args["use_plora"]:
        #     prefix_lora += "slora"
        # elif not args["use_slora"] and args["use_plora"]:
        #     prefix_lora += "plora"
        # else:
        #     prefix_lora += "slora+plora"
        # # rank=64_rho=0.7_slora+plora_mg=3.0_sg=0.5_pg=1.0
        # prefix_lora +=\
        #     "_mg=" + str(args["merge_gamma"]) +\
        #     "_sg=" + str(args["slora_gamma"]) +\
        #     "_pg=" + str(args["plora_gamma"]) # _mg:公式12里面的系数 _sg:共享分支缩放系数 (Shared-LoRA Scale Factor)  _pg:隔离分支缩放系数 (Particular / Isolated-LoRA Scale Factor)
    # rank=64_seq_lora
    logdir = 'logs/{}/{}_tasks/{}/{}'.format(args['dataset'], args['total_sessions'], prefix_head, prefix_lora)
    args['logdir'] = logdir # logs/ImageNet_R/20_tasks/standard_head/rank=64_seq_lora
    print(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}_slora:{}_plora:{}_rank:{}_{}_{}_{}-{}'.format(
        args['seed'], args["use_slora"], args["use_plora"], args['rank'],
        args.get("lora_type", "lora"), args['model_name'], args['optim'], args['lrate']))
    #logfilename = os.path.join(logdir, '{}_slora:{}_plora:{}_rank:{}_{}_{}_{}-{}'.format(args['seed'], args["use_slora"], args["use_plora"], args['rank'], args["lora_type"], args['model_name'], args['optim'], args['lrate']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[ # （双路输出处理器,把日志同时分发给两个目标
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
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
        logging.info('Time:{}'.format(time_end - time_start))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()

        if bool(args.get('dual_mask_track_w0_metrics', False)):
            w0_accuracy = model.eval_w0_task()
            if w0_accuracy is not None:
                w0_curve.append(round(w0_accuracy, 2))
                logging.info('W_pre-only NCM top1 curve: {}'.format(w0_curve))

        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
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
        if task_id > 0:
            diagonal = np.diag(model.acc_matrix)
            # Forgetting: 历史旧 Task 的历史最高准确率 - 当前测试准确率 的平均值
            forgetting = np.mean((np.max(model.acc_matrix, axis=1) -
                                model.acc_matrix[:, task_id])[:task_id])
            # BWT / 后向知识迁移: 当前测试准确率 - 刚学完该 Task 时的对角线准确率的平均值
            backward = np.mean((model.acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str = "Forgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
            logging.info(result_str)

    logging.info('Accuracy Matrix: \n {}'.format(model.acc_matrix.T.round(2)))
    logging.info('Average Accuracy: {}'.format(np.mean(cnn_curve['top1'])))
    logging.info('Last Accuracy: {}'.format(cnn_curve['top1'][-1]))

    if w0_curve:
        logging.info('W_pre-only NCM Average Accuracy: {}'.format(np.mean(w0_curve)))
        logging.info('W_pre-only NCM Last Accuracy: {}'.format(w0_curve[-1]))

    #
    # task_pred_curve = [round(acc * 100, 2) for acc in cnn_curve_task['top1']]
    # task_pred_avg = round(float(np.mean(cnn_curve_task['top1']) * 100.0), 2)
    # logging.info('Task Prediction Accuracy curve (%): {}'.format(task_pred_curve))
    logging.info('Task Prediction Accuracy average (%): {:.2f}'.format(task_pred_avg))
    # logging.info('Task Prediction Accuracy curve: {}'.format(cnn_curve_task['top1']))
    # logging.info('Final Task Prediction Accuracy: {:.2f}'.format(cnn_curve_task['top1'][-1]))
    logging.info( 'Final Task Prediction Accuracy (%): {:.2f}'.format( cnn_curve_task['top1'][-1] * 100.0))

# def _resolve_devices(device_values):
#     devices = []
#     for value in device_values:
#         if str(value) == '-1':
#             devices.append(torch.device('cpu'))
#         else:
#             devices.append(torch.device('cuda:{}'.format(value)))
#     return devices

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
    # args['device'] = _resolve_devices(args['device'])


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

