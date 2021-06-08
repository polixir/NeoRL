import os
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-on', '--output_name', type=str)
    parser.add_argument('-oa', '--ope_algo', type=str)
    parser.add_argument('-ng', '--num_gpus', type=float, default=1.0)
    parser.add_argument('--address', type=str, default=None)
    args = parser.parse_args()

    for domain in ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d', 'ib', 'finance', 'citylearn']:
        for level in ['low', 'medium', 'high']:
            for amount in [99, 999, 9999]:
                if domain == 'finance' and amount == 999: continue
                command = f'python scripts/launch_ope.py --domain {domain} --level {level} --amount {amount} -on {args.output_name} -oa {args.ope_algo} -ng {args.num_gpus}'
                if args.address is not None: command += f' --address {args.address}'
                os.system(command)
                time.sleep(10)