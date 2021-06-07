import os
import ray
import json
import time
import torch
import argparse
from ray import tune

import neorl
import offlinerl

from offlinerl.utils.exp import setup_seed
from d3pe.evaluator.fqe import FQEEvaluator
from d3pe.evaluator.IS import ISEvaluator
from d3pe.utils.data import get_neorl_datasets

ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
aim_folder = os.path.abspath(os.path.join(os.path.dirname(offlinerl.__file__), '..', 'offlinerl_tmp', '.aim'))
SEEDS = [7, 42, 210]

def check_file(domain, level, amount, algo, ope):
    ''' check if the result is already exist '''
    json_file = f'{domain}-{level}-{amount},{algo},{ope}.json'
    return json_file in os.listdir(ResultDir)

def launch_ope(config):
    ''' run on a seed '''
    setup_seed(config['seed'])
    if config['ope'] == 'fqe':
        evaluator = FQEEvaluator()
    elif config['ope'] == 'is':
        evaluator = ISEvaluator()
    
    train_dataset, val_dataset = get_neorl_datasets(config["domain"], config['level'], config['amount'])

    evaluator.initialize(train_dataset=train_dataset, val_dataset=val_dataset)

    exp_folder = os.path.join(config['task_folder'], config['exp_name'])
    with open(os.path.join(exp_folder, 'metric_logs.json'), 'r') as f:
        metrics = json.load(f)

    max_step = str(max(map(int, metrics.keys())))
    gt = metrics[max_step]['Reward_Mean_Env']

    policy_file = os.path.join(exp_folder, 'models', f'{max_step}.pt')
    policy = torch.load(policy_file)
    ope = evaluator(policy)

    return {
        'gt' : gt,
        'ope' : ope,
        'policy_file' : policy_file,
        'exp_name' : config['exp_name'],
        'seed' : config['seed'],
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help='select from `bc`, `bcq`, `plas`, `cql`, `crr`, `bremen`, `mopo`')
    parser.add_argument('--ope', type=str, help='select from `fqe` and `is`')
    parser.add_argument('--address', type=str, default=None, help='address of the ray cluster')
    args = parser.parse_args()

    ray.init(address=args.address)

    for task_name in sorted(filter(lambda x: args.algo in x, os.listdir(aim_folder))):
        task_folder = os.path.join(aim_folder, task_name)
        exp_names = list(filter(lambda x: not x == 'index' and not 'json' in x, os.listdir(task_folder)))

        split_name = task_name.split('-')
        if len(split_name) == 5:
            domain = split_name[0] + '-' + split_name[1]
            level = split_name[2]
            amount = int(split_name[3])
        else:
            domain = split_name[0]
            level = split_name[1]
            amount = int(split_name[2])

        if (not args.overwrite) and check_file(domain, level, amount, args.algo, args.ope): continue

        config = {
            'seed' : tune.grid_search(SEEDS),
            'ope' : args.ope,            
            'domain' : domain,
            'level' : level,
            'amount' : amount,
            'task_folder' : task_folder,
            'exp_name' : tune.grid_search(exp_names),
        }

        analysis = tune.run(
            launch_ope,
            name=f'{domain}-{level}-{amount}-{args.algo}-{args.ope}',
            config=config,
            queue_trials=True,
            metric='ope',
            mode='max',
            resources_per_trial={
                "cpu": 1,
                "gpu": 1.0,
            }
        )

        ''' process result '''
        df = analysis.results_df
        
        results = {seed : {} for seed in SEEDS}

        for i in range(len(df)):
            results[df['seed'][i]][df['exp_name'][i]] = {'gt' : df['gt'][i], 'ope' : df['ope'][i]}

        local_file = os.path.join(task_folder, f'{args.ope}.json')
        remote_file = os.path.join(ResultDir, f'{domain}-{level}-{amount},{args.algo},{args.ope}.json')
        with open(local_file, 'w') as f:
            json.dump(results, f, indent=4)
        os.system(f'cp {local_file} {remote_file}')

        time.sleep(20) # wait ray to release the resource          