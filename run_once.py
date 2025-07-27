# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/06/01 10:23
@File       :       run_once.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""
import os
import gc
import sys
import time
import random
import logging
import torch
import pynvml
import numpy as np
import pandas as pd

from cross_datasets import save_test_loader, load_test_loader
from models.AMIO import AMIO
from trains.ATIO import ATIO
from new_load_data import MMDataLoader
from config.config_regression import ConfigRegression
from config.config import *
from modelscope.hub.snapshot_download import snapshot_download

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args, dataloader):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    suffix = f'-mr{args.missing_rate[0]:.1f}-{args.seed}' if args.save_model else ''
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.datasetName}-{args.train_mode}{suffix}.pth')
    # indicate used gpu
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # add tmp tensor to increase the temporary consumption of GPU
    tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load models
    # dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    del tmp_tensor

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    results = atio.do_test(model, dataloader['test'], mode="TEST", batch=dataloader['test'].batch_size)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
 
    return results


def evaluation(args, dataloader):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    suffix = f'-mr{args.missing_rate[0]:.1f}-{args.seed}' if args.save_model else ''
    args.model_save_path = os.path.join(args.model_save_dir,
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}{suffix}.pth')
    # indicate used gpu
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load models
    # dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    del tmp_tensor

    # def count_parameters(model):
    #     answer = 0
    #     for p in model.parameters():
    #         if p.requires_grad:
    #             answer += p.numel()
    #             # print(p)
    #     return answer
    #
    # logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    atio = ATIO().getTrain(args)
    if args.datasetName == "mosi":
        try:
            test_model_save_path = "save_model/models/GQA/aligned/MOSI/best_model.pth"
            model.load_state_dict(torch.load(test_model_save_path))
        except FileNotFoundError:
            download_name = os.path.join(f"models/GQA/aligned/MOSI", 'best_model.pth')
            download_model_path = snapshot_download('Anony4Model/Parameters4HQAENet',
                                                      allow_patterns=download_name,
                                                      local_dir="save_model")
            test_model_save_path = os.path.join(download_model_path, download_name)
            model.load_state_dict(torch.load(test_model_save_path))
        # test_model_save_path = "save_model/models/GQA/aligned/MOSI/best_model.pth"  # TODO: Test for MOSI
    # test_model_save_path = "results/models/GQA/aligned/MOSI/MOSI2MOSEI_model.pth"  # TODO: Test for MOSI->MOSEI
    elif args.datasetName == "mosei":
        try:
            test_model_save_path = "save_model/models/GQA/aligned/MOSEI/best_model.pth"
            model.load_state_dict(torch.load(test_model_save_path))
        except FileNotFoundError:
            download_name = os.path.join(f"models/GQA/aligned/MOSEI", 'best_model.pth')
            download_model_path = snapshot_download('Anony4Model/Parameters4HQAENet',
                                                     allow_patterns=download_name,
                                                     local_dir="save_model")
            test_model_save_path = os.path.join(download_model_path, download_name)
            model.load_state_dict(torch.load(test_model_save_path))
        # test_model_save_path = "save_model/models/GQA/aligned/MOSEI/best_model.pth"  # TODO: Test for MOSEI
    # test_model_save_path = "results/models/GQA/aligned/MOSEI/MOSEI2MOSI_model.pth"  # TODO: Test for MOSEI->MOSI
    elif args.datasetName == "sims":
        try:
            test_model_save_path = "save_model/models/GQA/unaligned/sims/best_model.pth"
            model.load_state_dict(torch.load(test_model_save_path))
        except FileNotFoundError:
            download_name = os.path.join(f"models/GQA/unaligned/sims", 'best_model.pth')
            download_model_path = snapshot_download('Anony4Model/Parameters4HQAENet',
                                                     allow_patterns=download_name,
                                                     local_dir="save_model")
            test_model_save_path = os.path.join(download_model_path, download_name)
            model.load_state_dict(torch.load(test_model_save_path))
    else:
        assert os.path.exists(args.model_save_path)
        test_model_save_path = args.model_save_path
        model.load_state_dict(torch.load(test_model_save_path))

    # model.load_state_dict(torch.load(args.model_save_path))
    # model.load_state_dict(torch.load(test_model_save_path))
    print(f'Load model successfully: {test_model_save_path}')
    model.to(device)
    # do test
    # results = atio.do_test(model, dataloader['test'], mode="TEST", batch=dataloader['test'].batch_size)
    results = atio.do_test(model, dataloader, mode="TEST", batch=dataloader.batch_size)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    return results


def run_normal(args):
    res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    missing_rate = 0.0
    args = init_args
    # load config
    config = ConfigRegression(args)
    args = config.get_config()
    if args.do_evaluation:
        if args.datasetName == "mosi":
            testdataloader = load_test_loader("test_mosi.pkl")
        elif args.datasetName == "mosei":
            testdataloader = load_test_loader("test_mosei.pkl")
        else:
            testdataloader = load_test_loader("test_sims.pkl")

        test_results = evaluation(args, testdataloader)
        # restore results
        model_results.append(test_results)
        if args.datasetName == "sims":
            print('<<<Test>>> Min_MAE: {:.4f}, Best_Corr: {:.4f}, '
                  'Best_Acc5: {:.2f}, Best_Acc2: {:.2f}, '
                  'Best_F1:{:.2f},'.format(test_results["MAE"], test_results["Corr"],
                                                   (test_results["Mult_acc_5"] * 100),
                                                   (test_results["Mult_acc_2"] * 100),
                                                   (test_results["F1_score"] * 100)))
        else:
            print('<<<Test>>> Min_MAE: {:.4f}, Best_Corr: {:.4f}, Best_Acc7: {:.2f}, '
                  'Best_Acc5: {:.2f}, Best_Acc2: {:.2f}/{:.2f}, '
                  'Best_F1: {:.2f}/{:.2f},'.format(test_results["MAE"], test_results["Corr"],
                                                   (test_results["Mult_acc_7"] * 100),
                                                   (test_results["Mult_acc_5"] * 100), (test_results["Has0_acc_2"] * 100),
                                                   (test_results["Non0_acc_2"] * 100),
                                                   (test_results["Has0_F1_score"] * 100),
                                                   (test_results["Non0_F1_score"] * 100)))
        sys.exit(0)

    # load data
    dataloader = MMDataLoader(args)

    # run results
    for i, seed in enumerate(seeds):
        if i == 0 and args.data_missing:
            missing_rate = str(round(args.missing_rate[0], 1))
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s... with missing_rate=%s' %(args.modelName, missing_rate))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args, dataloader)
        # restore results
        model_results.append(test_results)
        logger.info(f"==> Test results of seed {seed}:\n{test_results}")
    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(res_save_dir, \
                        f'{args.datasetName}-{args.train_mode}-{missing_rate}.csv')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))
    # store results
    returned_res = res[1:]

    # detailed results
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(res_save_dir, \
                        f'{args.datasetName}-{args.train_mode}-{missing_rate}-detail.csv')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Time", "Model", "Params", "Seed"] + criterions)
    # seed
    for i, seed in enumerate(seeds):
        res = [cur_time, args.modelName, str(args), f'{seed}']
        for c in criterions:
            val = round(model_results[i][c]*100, 2)
            res.append(val)
        df.loc[len(df)] = res
    # mean
    res = [cur_time, args.modelName, str(args), '<mean/std>']
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    # max
    res = [cur_time, args.modelName, str(args), '<max/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        max_val = round(np.max(values)*100, 2)
        max_seed = seeds[np.argmax(values)]
        res.append((max_val, max_seed))
    df.loc[len(df)] = res
    # min
    res = [cur_time, args.modelName, str(args), '<min/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        min_val = round(np.min(values)*100, 2)
        min_seed = seeds[np.argmin(values)]
        res.append((min_val, min_seed))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Detailed results are added to %s...' %(save_path))

    return returned_res, criterions


def set_log(args):
    res_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    log_file_path = os.path.join(res_dir, f'run-once-{args.modelName}-{args.datasetName}.log')
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

if __name__ == '__main__':
    args = parse_args()
    global logger
    logger = set_log(args)
    args.seeds = [random.randint(0, 1000000) for _ in range(3)] if args.num_seeds is None else [
        random.randint(0, 1000000) for _ in range(args.num_seeds)]
    # args.seeds = [random.randint(295000, 300000) for _ in range(3)] if args.num_seeds is None else [
    #     random.randint(0, 1000000) for _ in range(args.num_seeds)]

    # args.seeds = [3407, 111, 1111, 11111, 111111] if args.num_seeds is None else list(
    #     range(args.num_seeds - 1, args.num_seeds + 1))
    # args.seeds = [111111] if args.num_seeds is None else list(
    #     range(args.num_seeds - 1, args.num_seeds + 1))
    # args.seeds = [111111, 1111111, 11111111] if args.num_seeds is None else list(range(args.num_seeds - 1, args.num_seeds + 1))
    # args.seeds.insert(0, 295029)
    # args.seeds.insert(0, 111111)
    args.num_seeds = len(args.seeds)

    if args.missing_rates is None:
        if args.datasetName in ['mosi', 'mosei']:
            args.missing_rates = np.arange(0, 1.0 + 0.1, 0.1).round(1)
        else:
            args.missing_rates = np.arange(0, 0.5 + 0.1, 0.1).round(1)
    else:
        args.missing_rates = np.arange(0, args.missing_rates + 0.1, 0.1).round(1)
        # args.missing_rates = np.arange(0.3, args.missing_rates + 0.1, 0.1).round(1)

    aggregated_results, metrics = [], []
    for mr in args.missing_rates:
        args.missing_rate = tuple([mr, mr, mr])
        res, criterions = run_normal(args)
        aggregated_results.append(res)
        metrics = criterions

    # save aggregated results
    save_path = os.path.join(args.res_save_dir, 'normals', \
                             f'{args.datasetName}-{args.train_mode}-aggregated.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model", "Missing_Rate"] + metrics)
    for mr, res in zip(args.missing_rates, aggregated_results):
        line = [args.modelName, mr] + res
        df.loc[len(df)] = line
    # auc
    agg_results = np.array(aggregated_results)[:,:,0]
    auc_res = np.sum(agg_results[:-1] + agg_results[1:], axis=0) / 2 * 0.1
    df.loc[len(df)] = [args.modelName, 'AUC'] + auc_res.round(1).tolist()
    df.to_csv(save_path, index=None)
    logger.info('Aggregated results are added to %s...' % (save_path))
