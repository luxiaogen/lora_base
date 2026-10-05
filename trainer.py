import copy
import logging
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from utils.data_manager import DataManager
from methods.dlora import Learner


def _set_random(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args["seed"])
        torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _set_device(args):
    args["device"] = [
        torch.device("cpu" if str(device) == "-1" else "cuda:" + str(device))
        for device in args["device"]
    ]


def train(args):
    seeds = copy.deepcopy(args["seed"])
    devices = args["device"].split(",")
    for seed in seeds:
        current = copy.deepcopy(args)
        current["seed"] = seed
        current["device"] = devices
        _train(current)


def _train(args):
    started = time.time()
    directory = os.path.join("logs", args["dataset"], "dualmask_baseline", args["prefix"])
    os.makedirs(directory, exist_ok=True)
    logfile = os.path.join(directory, str(args["seed"]) + ".log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s", force=True,
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    logging.info("Code revision: %s", revision)
    logging.info("Log: %s", os.path.abspath(logfile))
    _set_random(args)
    _set_device(args)
    for key, value in args.items():
        logging.info("%s: %s", key, value)
    data = DataManager(args["dataset"], args["shuffle"], args["seed"],
                       args["init_cls"], args["increment"], args)
    model = Learner(args)
    curve, old_curve, new_curve = [], [], []
    for task in range(int(args["max_tasks"])):
        model.incremental_train(data)
        accuracy = model.eval_task()
        model.after_task()
        curve.append(accuracy["top1"])
        if task > 0:
            old_curve.append(accuracy["grouped"]["old"])
            new_curve.append(accuracy["grouped"]["new"])
            forgetting = np.mean(
                (np.max(model.acc_matrix, axis=1) - model.acc_matrix[:, task])[:task])
            backward = np.mean(
                (model.acc_matrix[:, task] - np.diag(model.acc_matrix))[:task])
            logging.info("Forgetting: %.4f\tBackward: %.4f", forgetting, backward)
        logging.info("CNN: %s", accuracy["grouped"])
        logging.info("CNN top1 curve: %s", curve)
    logging.info("Accuracy Matrix:\n%s", model.acc_matrix.T.round(2))
    logging.info("Average Accuracy: %s", float(np.mean(curve)))
    logging.info("Last Accuracy: %s", curve[-1])
    if old_curve:
        logging.info("Task1+ mean Old: %.4f, New: %.4f", np.mean(old_curve), np.mean(new_curve))
    logging.info("Total experiment time: %.2f seconds", time.time() - started)
