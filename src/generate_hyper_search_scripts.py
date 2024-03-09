
# Author: Atharva Kulkarni
# File description: code to generate hparams

import os
import yaml
import argparse
from itertools import product
import inspect
import random
import numpy as np
import re



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="waterbirds",
        choices=['civilcomments', 'civilcomments_small', 'multinli', 'waterbirds', 'celeba'],
        help="The name of the dataset to use. Options: waterbirds, celeba, multinli, civilcomments, and civilcomments_small."
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="erm", 
        choices=["erm", "erm_l1", "erm_mt", "erm_mt_l1", "erm_mt2_l1", "suby", "subg", "rwy", "rwg", "dro", "jtt"],
        help="The method to use to train the models",
    )
    parser.add_argument(
        "--num_init_seeds", 
        type=int, 
        default=5, 
        help="Number of seeds to use.",
    )
    parser.add_argument(
        "--max_jobs", 
        type=int, 
        default=4, 
        help="Number number of jobs to run at one time.",
    )
    args = parser.parse_args()
    
    # Read the config file
    with open('./src/hparams.yaml', 'r') as file:
        config = yaml.safe_load(file)
    file.close()
    
    dataset_dir = os.path.join("./data", args.dataset)
    config = config[args.dataset][args.method]
    
    std_command = f"""
    python3 /src/train.py \
    --method {args.method} \
    --dataset {args.dataset} \
    --data_path {dataset_dir} \
    --num_train_epochs {config['num_train_epochs']} \
    --lr_scheduler_type {config['lr_scheduler_type']} \
    --output_dir models \
    --result_dir results \
    """

    # Add grid search hyperparameters
    if args.method in ["erm", "suby", "subg", "rwy", "rwg", "dro"]:
        commands = []
        # Add seed values   
        for seed in range(args.num_init_seeds):   
            command = ""
            for lr, wd, bs in list(product(config['learning_rate'], config['weight_decay'], config['bs'])):
                command = std_command + f"""--seed {seed} \
                --learning_rate {lr} \
                --weight_decay {wd} \
                --erm_batch_size {bs}
                """
                commands.append(inspect.cleandoc(command))
                
    elif args.method == "jtt":
        commands = []
        # Add seed values   
        for seed in range(args.num_init_seeds):   
            command = ""
            for lr, wd, bs, up, t in list(product(config['learning_rate'], config['weight_decay'], config['bs'], config['up'], config['T'])):
                command = std_command + f"""--seed {seed} \
                --learning_rate {lr} \
                --weight_decay {wd} \
                --erm_batch_size {bs} \
                --up {up} \
                --T {t}
                 """
                commands.append(inspect.cleandoc(command))   
                
    elif "mt" in args.method and "l1" not in args.method:
        lr = config['learning_rate'][0]
        wd = config['weight_decay'][0]
        commands = []
        for seed in range(args.num_init_seeds):
            command = ""
            for bs, erm_weight, mt_weight in list(product(config['bs'], config['erm_weight'], config['mt_weight'])):
                command = std_command + f"""--seed {seed} \
                --learning_rate {lr} \
                --weight_decay {wd} \
                --erm_batch_size {bs} \
                --mt_batch_size {bs} \
                --erm_weight {round(np.exp(erm_weight), 4)} \
                --mt_weight {round(np.exp(mt_weight), 4)}
                 """
                commands.append(inspect.cleandoc(command))
        
    
    elif "mt" not in args.method and "l1" in args.method:
        lr = config['learning_rate'][0]
        wd = config['weight_decay'][0]
        commands = []
        for seed in range(args.num_init_seeds):
            command = ""
            for bs, erm_weight, reg_weight in list(product(config['bs'], config['erm_weight'], config['reg_weight'])):
                command = std_command + f"""--seed {seed} \
                --learning_rate {lr} \
                --weight_decay {wd} \
                --erm_batch_size {bs} \
                --mt_batch_size {bs} \
                --erm_weight {round(np.exp(erm_weight), 4)} \
                --reg_weight {round(np.exp(reg_weight), 4)}
                 """
                commands.append(inspect.cleandoc(command))
       
       
    elif "mt" in args.method and "l1" in args.method:
        lr = config['learning_rate'][0]
        wd = config['weight_decay'][0]
        commands = []
        for seed in range(args.num_init_seeds):
            command = ""
            for bs, erm_weight, mt_weight, reg_weight in list(product(config['bs'], config['erm_weight'], config['mt_weight'], config['reg_weight'])):
                command = std_command + f"""--seed {seed} \
                --learning_rate {lr} \
                --weight_decay {wd} \
                --erm_batch_size {bs} \
                --mt_batch_size {bs} \
                --erm_weight {round(np.exp(erm_weight), 4)} \
                --mt_weight {round(np.exp(mt_weight), 4)} \
                --reg_weight {round(np.exp(reg_weight), 4)}
                 """
                commands.append(inspect.cleandoc(command))
       
       
    print(f"\nTotal scripts to run: {len(commands)}\n")
    
    log_dir = os.path.join("./logs", args.method, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    error_dir = os.path.join("./errors", args.method, args.dataset)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    
    if not os.path.exists("./hparams_files"):
        os.makedirs("./hparams_files")
        
    fname = f"hparams_files/{args.dataset}_{args.method}.txt"
    
    with open(fname, "w") as f:
        for command in commands:
            f.write("%s" % inspect.cleandoc(command.lstrip()))
    f.close()
    
    params_args = "params=$(tail -n+${SLURM_ARRAY_TASK_ID} " + f"{fname} | head -n1)"

    if args.dataset in ['waterbirds', 'celeba']:
        exp_name = "vit_wga_hp_exp"
    else:
        exp_name = "bert_wga_hp_exp"
        
        
    job_file_header = inspect.cleandoc(
        f"""
        #!/bin/bash
        
        #SBATCH --job-name={exp_name}
        #SBATCH --array=1-{len(commands)}%{args.max_jobs}
        #SBATCH --partition=babel-shared-long
        #SBATCH --mem=50GB
        #SBATCH --time=2-23:00:00
        #SBATCH --gres gpu:3090:1
        #SBATCH --output={log_dir}/{args.dataset}_array_job_%A_%a.log
        #SBATCH --error={error_dir}/{args.dataset}_array_job_%A_%a.err

        {params_args}   
        $params
        """
    )
    
    if not os.path.exists("./scripts"):
        os.makedirs("./scripts")
        
    bash_file = f"train_{args.dataset}_{args.method}_hp.sh"
    
    with open(os.path.join("./scripts", bash_file), "w") as f:
        f.write("%s\n" % job_file_header)
    f.close()
    
    print('Done')