import os
import argparse
import logging
import sys
from pathlib import Path
import pprint
from sympy import Segment
import yaml
import numpy as np
import torch.distributed as dist
from torch import multiprocessing


import utils,configs,datasets
import torch_utils.pipelines,torch_utils.models

import torch.multiprocessing as mp
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-c', '--cfg_file', help='path to the config file',
                        default='/home/zxhc/Workspace/Open3D-ML/configs/config.yaml')
    parser.add_argument('--device',
                        help='devices to run the pipeline, cpu or cuda',
                        default='cuda')
    parser.add_argument('--device_ids',
                        nargs='+',
                        help='cuda device list',
                        default=[0])
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--nodes', help='number of nodes', default=1, type=int)
    parser.add_argument('--node_rank',
                        help='ranking within the nodes, default: 0. To get from'
                        ' the environment, enter the name of an env var eg: '
                        '"SLURM_NODEID".',
                        default="0",
                        type=str)
    parser.add_argument(
        '--host',
        help='Host for distributed training, default: localhost',
        default='localhost')
    parser.add_argument('--port',
                        help='port for distributed training, default: 12355',
                        default='12355')
    parser.add_argument(
        '--backend',
        help=
        'backend for distributed training. One of (nccl, gloo)}, default: gloo',
        default='gloo')

    args = parser.parse_args()
    try:
        args.node_rank = int(args.node_rank)
    except ValueError:  # str => get from environment
        args.node_rank = int(os.environ[args.node_rank])

    print("regular arguments")
    print(yaml.dump(vars(args)))

    return args


def main():
    cmd_line = ' '.join(sys.argv[:])
    args = parse_args()
    
    #获取随机数生成器
    rng = np.random.default_rng(seed=0)
        
    #加载配置文件
    cfg = utils.Config.load_from_file(args.cfg_file)

    Pipeline = torch_utils.pipelines.SemanticSegmentation
    Model = torch_utils.models.RandLANet
    Dataset = datasets.Custom3D

    #获取各个模块的配置字典
    cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = cfg.dataset, cfg.pipeline, cfg.model

    cfg_dict_dataset['seed'] = rng
    cfg_dict_model['seed'] = rng
    cfg_dict_pipeline['seed'] = rng

    cfg_dict_pipeline["device"] = args.device
    cfg_dict_pipeline["device_ids"] = args.device_ids


    cfg_tb = {
        'cmd_line': cmd_line,
        'dataset': pprint.pformat(cfg_dict_dataset, indent=2),
        'model': pprint.pformat(cfg_dict_model, indent=2),
        'pipeline': pprint.pformat(cfg_dict_pipeline, indent=2)
    }
    args.cfg_tb = cfg_tb
    args.distributed = args.device != 'cpu' and len(
        args.device_ids) > 1

    if not args.distributed:
        # 单GPU训练模式
        dataset = Dataset(**cfg_dict_dataset)
        model = Model(**cfg_dict_model)
        pipeline = Pipeline(model, dataset, **cfg_dict_pipeline)

        pipeline.cfg_tb = cfg_tb

        if cfg_dict_pipeline['split'] == 'test':
            pipeline.run_test()
        else:
            pipeline.run_train()

    else:
        # 多GPU训练模式
        mp.spawn(main_worker,
                 args=(Dataset, Model, Pipeline, cfg_dict_dataset,
                       cfg_dict_model, cfg_dict_pipeline, args),
                 nprocs=len(args.device_ids))


def setup(rank, world_size, args):
    os.environ['PRIMARY_ADDR'] = args.host
    os.environ['PRIMARY_PORT'] = args.port

    # initialize the process group
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(local_rank, Dataset, Model, Pipeline, cfg_dict_dataset,
                cfg_dict_model, cfg_dict_pipeline, args):
    rank = args.node_rank * len(args.device_ids) + local_rank
    world_size = args.nodes * len(args.device_ids)
    setup(rank, world_size, args)

    cfg_dict_dataset['rank'] = rank
    cfg_dict_model['rank'] = rank
    cfg_dict_pipeline['rank'] = rank

    rng = np.random.default_rng(rank)
    cfg_dict_dataset['seed'] = rng
    cfg_dict_model['seed'] = rng
    cfg_dict_pipeline['seed'] = rng

    device = f"cuda:{args.device_ids[local_rank]}"
    print(
        f"local_rank = {local_rank}, rank = {rank}, world_size = {world_size},"
        f" gpu = {device}")

    cfg_dict_model['device'] = device
    cfg_dict_pipeline['device'] = device

    dataset = Dataset(**cfg_dict_dataset)
    model = Model(**cfg_dict_model, mode=args.mode)
    pipeline = Pipeline(model,
                        dataset,
                        distributed=args.distributed,
                        **cfg_dict_pipeline)

    pipeline.cfg_tb = args.cfg_tb

    if args.split == 'test':
        if rank == 0:
            pipeline.run_test()
    else:
        pipeline.run_train()

    cleanup()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    multiprocessing.set_start_method('forkserver')
    sys.exit(main())
