import neorl

if __name__ == '__main__':
    for domain in ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'ib', 'finance', 'citylearn']:
        for level in ['low', 'medium', 'high']:
            for amount in [100, 1000, 10000] if not domain == 'finance' else [100, 1000]:
                env = neorl.make(domain)
                env.get_dataset(data_type=level, train_num=amount)