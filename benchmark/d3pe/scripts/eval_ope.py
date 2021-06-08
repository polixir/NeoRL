import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from d3pe.metric.score import RC_score, TopK_score, get_policy_mean

BenchmarkFolder = 'benchmarks'
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', type=str)
    parser.add_argument('-l', '--level', type=str)
    parser.add_argument('-a', '--amount', type=int)
    parser.add_argument('-en', '--evaluate_name', type=str)
    args = parser.parse_args()

    if not os.path.exists('ope_plots'): os.makedirs('ope_plots')

    ''' load ope and gt values '''
    task = f'{args.domain}-{args.level}-{args.amount}'
    task_folder = os.path.join(BenchmarkFolder, task)
    evaluate_file = os.path.join(task_folder, args.evaluate_name + '.json')
    with open(evaluate_file, 'r') as f:
        evaluate_json = json.load(f)
    
    gt_file = os.path.join(task_folder, 'gt.json')
    with open(gt_file, 'r') as f:
        gt_json = json.load(f)

    real_score = []
    esitmate_score = []
    for key in gt_json.keys():
        real_score.append(gt_json[key])
        esitmate_score.append(evaluate_json[key])
    
    ''' evaluation '''
    print('RC score:', RC_score(real_score, esitmate_score))
    print('Top 1 mean:', TopK_score(real_score, esitmate_score, 1, 'mean'))
    print('Top 3 mean:', TopK_score(real_score, esitmate_score, 3, 'mean'))
    print('Top 5 mean:', TopK_score(real_score, esitmate_score, 5, 'mean'))
    print('Top 1 max:', TopK_score(real_score, esitmate_score, 1, 'max'))
    print('Top 3 max:', TopK_score(real_score, esitmate_score, 3, 'max'))
    print('Top 5 max:', TopK_score(real_score, esitmate_score, 5, 'max'))
    print('Policy Mean Score:', get_policy_mean(real_score))

    ''' plot '''
    plt.figure()
    gt_scale = 1 / (1 - 0.99) / 1000.0 if not 'finance' in task else 1 / (1 - 0.99) / 2516
    try:
        r = pearsonr(np.array(real_score) * gt_scale, np.array(esitmate_score))[0]
    except:
        r = float('nan')
    plt.scatter(np.array(real_score) * gt_scale, np.array(esitmate_score))
    max_value = max((np.array(real_score) * gt_scale).max(), np.array(esitmate_score).max())
    min_value = min((np.array(real_score) * gt_scale).min(), np.array(esitmate_score).min())
    plt.plot([min_value, max_value], [min_value, max_value], 'k--')
    plt.title(task + ' (r=%.2f)' % r)
    plt.xlabel('Ground Truth', {'size' : 12})
    plt.ylabel(args.evaluate_name.upper(), {'size' : 12})
    plt.savefig(f'ope_plots/{task}-{args.evaluate_name}.png')