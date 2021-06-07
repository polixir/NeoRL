import os
import ray
import json
import time
import argparse
import numpy as np
from ray import tune

from offlinerl.algo import algo_select
from offlinerl.data import load_data_from_neorl
from offlinerl.evaluation import OnlineCallBackFunction, PeriodicCallBack

SEEDS = [7, 42, 210]

ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

def training_function(config):
    ''' run on a seed '''
    config["kwargs"]['seed'] = config['seed']
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])
    train_buffer, val_buffer = load_data_from_neorl(algo_config["task"], algo_config["task_data_type"], algo_config["task_train_num"])
    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_config['dynamics_path'] = os.path.join(config['dynamics_root'], 
        f'{algo_config["task"]}-{algo_config["task_data_type"]}-{algo_config["task_train_num"]}-{config["seed"]}.pt')
    algo_config['behavior_path'] = os.path.join(config['behavior_root'], 
        f'{algo_config["task"]}-{algo_config["task_data_type"]}-{algo_config["task_train_num"]}-{config["seed"]}.pt')
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = PeriodicCallBack(OnlineCallBackFunction(), 50)
    callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"], number_of_runs=1000)

    algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)
    algo_trainer.exp_logger.flush()
    time.sleep(10) # sleep ensure the log is flushed even if the disks or cpus are busy 

    result, parameter = find_result(algo_trainer.index_path)

    return {
        'reward' : result,
        'parameter' : parameter,
        'seed' : config['seed'],
    }

def upload_result(task_name : str, algo_name : str, results : list):
    ''' upload the result '''
    # upload txt
    file_name = task_name + ',' + algo_name + '.txt'
    reward_means = [result['reward_mean'] for result in results]
    max_reward_mean = max(reward_means)
    best_index = reward_means.index(max_reward_mean)
    best_result = results[best_index]
    with open(os.path.join(ResultDir, file_name), 'w') as f:
        f.write(str(best_result['reward_mean']) + '+-' + str(best_result['reward_std']))
        for k, v in best_result['parameter'].items():
            f.write('\n')
            f.write(f'{k} : {v}')

    # upload json
    file_name = task_name + ',' + algo_name + '.json'
    with open(os.path.join(ResultDir, file_name), 'w') as f:
        json.dump(results, f, indent=4)

def find_result(exp_dir : str):
    ''' return the online performance of last epoch and the hyperparameter '''
    data_file = os.path.join(exp_dir, 'objects', 'map', 'dictionary.log')
    with open(data_file, 'r') as f:
        data = json.load(f)
    result = data['__METRICS__']['Reward_Mean_Env'][0]['values']['last']
    grid_search_keys = list(data['hparams']['grid_tune'].keys())
    parameter = {k : data['hparams'][k] for k in grid_search_keys}
    return result, parameter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--level', type=str)
    parser.add_argument('--amount', type=int)
    parser.add_argument('--algo', type=str, help='select from `bc`, `bcq`, `plas`, `cql`, `crr`, `bremen`, `mopo`')
    parser.add_argument('--address', type=str, default=None, help='address of the ray cluster')
    args = parser.parse_args()

    ray.init(args.address)

    domain = args.domain
    level = args.level
    amount = args.amount
    algo = args.algo

    ''' run and upload result '''
    config = {}
    config["kwargs"] = {
        "exp_name" : f'{domain}-{level}-{amount}-{algo}',
        "algo_name" : algo,
        "task" : domain,
        "task_data_type" : level,
        "task_train_num" : amount,
    }
    _, _, algo_config = algo_select({"algo_name" : algo})

    parameter_names = []
    grid_tune = algo_config["grid_tune"]
    for k, v in grid_tune.items():
        parameter_names.append(k)
        config[k] = tune.grid_search(v)

    config['seed'] = tune.grid_search(SEEDS)
    config['dynamics_root'] = os.path.abspath('dynamics')
    config['behavior_root'] = os.path.abspath('behaviors')
    
    analysis = tune.run(
        training_function,
        name=f'{domain}-{level}-{amount}-{algo}',
        config=config,
        queue_trials=True,
        metric='reward',
        mode='max',
        resources_per_trial={
            "cpu": 1,
            "gpu": 1.0,
        }
    )

    df = analysis.results_df

    ''' process result '''
    results = {}
    for i in range(len(df)):
        parameter = {}
        for pn in parameter_names:
            parameter[pn] = df[f'parameter.{pn}'][i]
            if type(parameter[pn]) == np.int64:
                parameter[pn] = int(parameter[pn]) # covert to python type
        parameter_string = str(parameter)

        if not parameter_string in results:
            results[parameter_string] = {
                'parameter' : parameter,
                'rewards' : [0, 0, 0],
            }

        results[parameter_string]['rewards'][SEEDS.index(df['seed'][i])] = df['reward'][i]

    def summary_result(single_result):
        single_result.update({
            'reward_mean' : np.mean(single_result['rewards']),
            'reward_std' : np.std(single_result['rewards']),
        })
        return single_result

    results = [summary_result(single_result) for single_result in results.values()]
        

    ''' upload result '''
    upload_result(f'{domain}-{level}-{amount}', algo, results)