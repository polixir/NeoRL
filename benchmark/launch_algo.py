import os
import time
import argparse

ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

def check_file(domain, level, amount, algo):
    ''' check if the result is already exist '''
    json_file = f'{domain}-{level}-{amount},{algo}.json'
    return json_file in os.listdir(ResultDir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help='select from `bc`, `bcq`, `plas`, `cql`, `crr`, `bremen`, `mopo`')
    parser.add_argument('--address', type=str, default=None, help='address of the ray cluster')
    parser.add_argument('--domain', type=str, default=None, help='domain of tasks')
    parser.add_argument('--level', type=str, default=None, help='level/quality of dataset. Should be one of "low", "medium", "high" or "human". Note "human" is for sales promotion only.')
    parser.add_argument('--amount', type=int, default=None, help='number of trajectories')
    args = parser.parse_args()

    if not os.path.exists(ResultDir): os.makedirs(ResultDir)

    if args.domain is None:
        ''' run a single algorithm on all the tasks '''
        domains = ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'ib', 'finance', 'citylearn', 'sp']
    else:
        domains = [args.domain]

    for domain in domains:
        if args.level is None:
            data_level = ['low', 'medium', 'high']
        else:
            data_level = [args.level]
        if domain == 'sp':
            data_level = ['human']
        for level in data_level:
            if args.amount is not None:
                amounts = [args.amount]
            else:
                if domain == 'finance':
                    amounts = [100, 1000]
                elif domain == 'sp':
                    amounts = [10000]
                else:
                    amounts = [100, 1000, 10000]
            for amount in amounts:
                if not check_file(domain, level, amount, args.algo):
                    if args.address is not None:
                        os.system(f'python launch_task.py --domain {domain} --level {level} --amount {amount} --algo {args.algo} --address {args.address}')
                    else:
                        os.system(f'python launch_task.py --domain {domain} --level {level} --amount {amount} --algo {args.algo}')
                    time.sleep(20)  # wait ray to release the resource