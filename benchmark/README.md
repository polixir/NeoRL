## Instruction for reproducing the benchmark
Before you start, make sure you have followed the instruction [here](../README.md) to install NeoRL. All the results of the benchmark will be stored in the `results` folder.
**Update:** We have also shared the raw result in [`results`](results/) (including the newly added sales promotion results), and you are free to analyze the result on your own.
### Step 1: Install dependencies
```bash
git submodule init
git submodule update

cd OfflineRL/
pip install -e .
mkdir offlinerl_tmp
cd offlinerl_tmp
mkdir offlinerl_datasets
aim init
cd ../..

cd d3pe/
pip install -e .
cd ..
```

### Step 2: Download the datasets
Download all the datasets before training the algorithms by `python download_datasets.py`.

### Step 3: Train policies by Offline RL Algorithms
You can use `launch_algo.py` for this part. The script will automatically launch an algorithm for 51 tasks provided in NeoRL, and training them in parallel based on `ray`. For example, you can use `python launch_algo.py --algo bc` to launch benchmark for BC algorithm. The trained policies will be stored in `OfflineRL/offlinerl_tmp/.aim`. 
The *sales promotion* environment (sp) used human expert datasets, and we do not unify it now as the Gym-MuJoCo, FinRL, CityLearn and IB environments (can be handled with a few lines of codes).
So to run BC on the *sales promotions* task, you should use `launch_algo.py --domain sp --algo bc` to run BC on sp environment.

Note that, to speed up the benchmark, for model-based algorithms, i.e. BREMEN and MOPO, we used pretrained dynamic models. These models can be obtained by `python pretrain_dynamics.py`. You should indicate the domain (The *sales promotion* environment used human expert data, thus it needs manually switch) for pretraining models.

For BREMEN, we also used the pretrained behavior policy obtained by BC algorithm. Thus, before run benchmark on BREMEN, make sure the BC benchmark is done, and export the policies by `python export_bc.py`.

### Step 4: Evaluate the trained policies by OPEs
You can use `launch_ope.py` for this part. The script will automatically launch an OPE algorithms to evaluate all the policies trained by the designated offline RL algorithm. For example, you can use `python launch_ope.py --algo bc --ope fqe` to launch FQE algorithm to evaluate all the policies trained by BC.

