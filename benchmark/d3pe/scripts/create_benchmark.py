'''
    This file is used to choose models for the benchmark.
    Original models are obtained by the NeoRL experiment.

    For each task (<domain>-<level>-<amount>), we select 
    32 models for the benchmark with the following rules:

    1. Evenly choose 17 models from the range of the values.
    2. Uniformly choose 15 models from the remain models.
'''

import os
import json
import numpy as np
from collections import OrderedDict

BehaviorPerformance = {
    'HalfCheetah-v3-low' : 3195,
    'HalfCheetah-v3-medium' : 6027,
    'HalfCheetah-v3-high' : 9020,
    'Hopper-v3-low' : 508,
    'Hopper-v3-medium' : 1530,
    'Hopper-v3-high' : 2294,
    'Walker2d-v3-low' : 1572,
    'Walker2d-v3-meidum' : 2547,
    'Walker2d-v3-high' : 3550,
    'ib-low' : -344311,
    'ib-medium' : -283121,
    'ib-high' : -220156,
    'finance-low' : 150,
    'finance-medium' : 300,
    'finance-high' : 441,
    'citylearn-low' : 28500,
    'citylearn-medium' : 37800,
    'citylearn-high' : 48600,
}

def find_nearest_key(dic, value):
    key = None
    diff = float('inf')
    for k, v in dic.items():
        _diff = np.abs(v - value)
        if _diff < diff:
            diff = _diff
            key = k
    return key

if __name__ == '__main__':
    models_folder = 'ope_models'
    assert os.path.exists(models_folder), "Please download the base models!"
    output_folder = 'benchmarks'
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    np.random.seed(42)
    
    for task_domain in ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'ib', 'finance', 'citylearn']:
        for task_level in ['low', 'medium', 'high']:
            for task_amount in [99, 999, 9999] if not task_domain == 'finance' else [99, 999]:
                task = f'{task_domain}-{task_level}-{task_amount}'
                print(f'Creating benchmark for {task}')
                task_folders = [os.path.join(models_folder, task_folder) for task_folder in filter(lambda x: task + '-' in x, os.listdir(models_folder))]
                models_and_gt = OrderedDict()
                for task_folder in task_folders:
                    for exp_folder in [os.path.join(task_folder, exp_name) for exp_name in os.listdir(task_folder)]:
                        with open(os.path.join(exp_folder, 'gt.json'), 'r') as f:
                            gts = json.load(f)
                        for model_name, gt in gts.items():
                            models_and_gt[os.path.join(exp_folder, 'models', model_name)] = gt

                if not os.path.exists(os.path.join(output_folder, task, 'models')): os.makedirs(os.path.join(output_folder, task, 'models'))
                model_index = 0

                gt = np.array(list(models_and_gt.values()))
                max_gt = np.max(gt)
                min_gt = np.min(gt)

                print('Performance Min:', min_gt)
                print('Performance Max:', max_gt)

                output_gt = {}

                performance_to_choose = np.linspace(min_gt, max_gt, num=17)

                for performance in performance_to_choose:
                    key = find_nearest_key(models_and_gt, performance)
                    value = models_and_gt.pop(key)
                    save_path = os.path.join(output_folder, task, 'models', f'model_{model_index}_{value}.pt')
                    model_index += 1
                    os.system(f'cp {key} {save_path}')      
                    output_gt[save_path] = value  

                for key in np.random.choice(list(models_and_gt.keys()), size=15, replace=False):
                    value = models_and_gt.pop(key)
                    save_path = os.path.join(output_folder, task, 'models', f'model_{model_index}_{value}.pt')
                    model_index += 1
                    os.system(f'cp {key} {save_path}')      
                    output_gt[save_path] = value  

                with open(os.path.join(output_folder, task, 'gt.json'), 'w') as f:
                    json.dump(output_gt, f, indent=4)
                            
