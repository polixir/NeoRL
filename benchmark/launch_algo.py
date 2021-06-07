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
    args = parser.parse_args()

    if not os.path.exists(ResultDir): os.makedirs(ResultDir)

    ''' run a single algorithm on all the tasks '''
    tasks = []
    for domain in ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'ib', 'finance', 'citylearn']:
        for level in ['low', 'medium', 'high']:
            for amount in [100, 1000, 10000] if not domain == 'finance' else [100, 1000]:
                if not check_file(domain, level, amount, args.algo):
                    if args.address is not None:
                        os.system(f'python launch_task.py --domain {domain} --level {level} --amount {amount} --algo {args.algo} --address {args.address}')
                    else:
                        os.system(f'python launch_task.py --domain {domain} --level {level} --amount {amount} --algo {args.algo}')
                    time.sleep(20) # wait ray to release the resource