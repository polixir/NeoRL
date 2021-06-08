import os
import ray
import json
import neorl
import torch
import argparse
from tqdm import tqdm

from d3pe.utils.func import get_evaluator_by_name
from d3pe.utils.data import get_neorl_datasets

BenchmarkFolder = 'benchmarks'

def get_ope(policy, ope_algo : str, task_name : str, task_level : str, task_amount : int):
    evaluator = get_evaluator_by_name(ope_algo)()
    train_dataset, val_dataset = get_neorl_datasets(task_name, task_level, task_amount)
    evaluator.initialize(train_dataset, val_dataset, task=task_name)
    return evaluator(policy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--level', type=str)
    parser.add_argument('--amount', type=int)
    parser.add_argument('-on', '--output_name', type=str)
    parser.add_argument('-oa', '--ope_algo', type=str)
    parser.add_argument('-ng', '--num_gpus', type=float, default=1.0)
    parser.add_argument('--address', type=str, default=None)
    parser.add_argument('-f', '--force', action='store_true', help='force means overwrite the existing results')
    args = parser.parse_args()

    # start or attach ray cluster
    ray.init(args.address)

    # make sure the data exist
    env = neorl.make(args.domain)
    env.get_dataset(data_type=args.level, train_num=args.amount)
    
    # create and load file
    task = f'{args.domain}-{args.level}-{args.amount}'
    task_folder = os.path.join(BenchmarkFolder, task)
    output_file = os.path.join(task_folder, args.output_name + '.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output_json = json.load(f)
    else:
        output_json = {}
    
    gt_file = os.path.join(task_folder, 'gt.json')
    with open(gt_file, 'r') as f:
        gt_json = json.load(f)

    # launch jobs
    pending_jobs = {}
    ope_function = ray.remote(num_gpus=args.num_gpus)(get_ope)
    for policy_file in gt_json.keys():
        if args.force or not policy_file in output_json.keys():
            policy = torch.load(policy_file, map_location='cpu')
            pending_jobs[ope_function.remote(policy, args.ope_algo, args.domain, args.level, args.amount)] = policy_file
    
    # collecting results
    timer = tqdm(total=len(pending_jobs))
    while len(pending_jobs) > 0:
        finished_jobs, unfinished_jobs = ray.wait(list(pending_jobs.keys()))
        if len(finished_jobs) > 0:
            finished_job = finished_jobs[0]
            policy_file = pending_jobs.pop(finished_job)
            value = ray.get(finished_job)
            output_json[policy_file] = value
            # immediately write out the new result
            with open(output_file, 'w') as f:
                json.dump(output_json, f, indent=4)
            timer.update(1)