import os
import json
import offlinerl

aim_folder = os.path.abspath(os.path.join(os.path.dirname(offlinerl.__file__), '..', 'offlinerl_tmp', '.aim'))
target_folder = 'behaviors'

if not os.path.exists(target_folder): os.makedirs(target_folder)

for task_name in sorted(filter(lambda x: 'bc' in x, os.listdir(aim_folder))):
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

    for exp_name in exp_names:
        exp_folder = os.path.join(task_folder, exp_name)
        
        with open(os.path.join(exp_folder, 'objects', 'map', 'dictionary.log'), 'r') as f:
            data = json.load(f)
            seed = data['hparams']['seed']

        with open(os.path.join(exp_folder, 'metric_logs.json'), 'r') as f:
            metrics = json.load(f)

        max_step = max([int(name.split('.')[0]) for name in os.listdir(os.path.join(exp_folder, 'models'))])
        
        policy_file = os.path.join(exp_folder, 'models', f'{max_step}.pt')
        target_policy_file = os.path.join(target_folder, f'{domain}-{level}-{amount}-{seed}.pt')

        os.system(f"cp {policy_file} {target_policy_file}")                 