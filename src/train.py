# Author: Atharva Kulkarni
# File description: code to train model for improving worst group performance.

import gc
import os
import sys
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from transformers import SchedulerType
import models_mt as models
from prepare_datasets import get_loaders
from datetime import datetime



def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument(
        "--seed",
         type=int, 
         default=0, 
         help="A seed for reproducible training."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="civilcomments",
        choices=['civilcomments', 'civilcomments_small', 'multinli', 'waterbirds', 'celeba'],
        help="The name of the dataset to use. Options: waterbirds, celeba, multinli, civilcomments."
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="erm", 
        choices=["erm", "erm_l1","erm_mt", "erm_mt_l1", "erm_mt2", "erm_mt2_l1", "tapt", "suby", "subg", "rwy", "rwg", "dro", "jtt"],
        help="The method to use to train the models",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../data/waterbirds/",
        help="The path to the dataset directory."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=10, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--erm_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the ERM training dataloader.",
    )
    parser.add_argument(
        "--mt_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the MultiTask training dataloader.",
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mt_grad_acc",
        type=int,
        default=1,
        help="Number of updates steps to accumulate MultiTask performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=None, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--erm_weight", 
        type=float, 
        default=1.0,
        help="Sacling factor for erm loss. To be used only when the method is erm_mt",
    )
    parser.add_argument(
        "--mt_weight", 
        type=float, 
        default=1.0,
        help="Sacling factor for multi-tak loss. To be used only when the method is erm_mt",
    )
    parser.add_argument(
        "--reg_weight", 
        type=float, 
        default=1.0,
        help="Sacling factor for l1 loss. To be used only when the method is erm_mt",
    )
    parser.add_argument(
        "--reset_loader_every_epoch", 
        action="store_true", 
        help="Whether to reset dataloaders after each epoch.",
    )
    parser.add_argument(
        "--T", 
        type=int,
         default=2,
         help="Number of epochs to train the first pass of JTT"
    )
    parser.add_argument(
        "--up", 
        type=int, 
        default=20,
        help="Upweighing parameter for JTT."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='./models/', 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default='./results/', 
        help="Where to store the prediction results."
    )
            
    return parser.parse_args()


def save_results(results_dict, arguments, params_str, epoch):
    for data_type in results_dict:
        preds, gold, group_labels = results_dict[data_type]['preds'], results_dict[data_type]['gold'], results_dict[data_type]['group_labels']
        df = pd.DataFrame(list(zip(preds, gold, group_labels)), columns=['preds', 'gold', 'group_labels'])               
            
        append = ""       
        if 'small' in arguments.data_path:
            append="_small"
        result_dir = os.path.join(arguments.result_dir, f"{arguments.dataset}{append}/{data_type}")
        
        result_dir = result_dir.replace("erm_mt_l1", "erm_mt_l2")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        file_name = os.path.join(result_dir, f"{arguments.dataset}{append}_{data_type}_{arguments.method}_{params_str}seed_{arguments.seed}_best.csv")
       
        df.to_csv(file_name, index=False)

                
def run_experiment(args):
    
    # ------------------------------------- Helper Functions -------------------------------------
    
    def my_get_loader():
        mt_loader = None
        if "mt" in args.method:
            loaders = get_loaders(
                args.data_path, 
                args.dataset, 
                [args.erm_batch_size, args.mt_batch_size],
                args.method,
                seed=args.seed,
            )
            erm_loader = loaders["tr"][0]
            print(f"mt_tr size: {len(loaders['tr'][1].dataset)}\tmt_tr dataloader size: {len(loaders['tr'][1])}")
            mt_loader = iter(loaders["tr"][1])
            base_dataset = erm_loader.dataset
        else:
            loaders = get_loaders(
                args.data_path, 
                args.dataset, 
                args.erm_batch_size,
                args.method,
                seed=args.seed,
            )
            erm_loader = loaders["tr"]
        
        return erm_loader, mt_loader, loaders["va"], loaders["te"]
    
    # Check if next batch for mt exists. If not create new loader
    def try_get_mt_batch(loader_):
        try:
            mt_batch = next(loader_)
            return mt_batch, loader_
        except:
            print("\nReloading MT data\n")
            del loader_
            gc.collect()
            _, loader_, _, _ = my_get_loader()
            mt_batch = next(loader_)
            return mt_batch, loader_

    # --------------------------------------------------------------------------------------------

    print("This process has the PID: ", os.getpid())
            
    print("\nArguments: \n")
    for key, val in vars(args).items():
        print(f"{key}:\t{val}")

    if args.method in ["erm", "suby", "subg", "rwy", "rwg", "dro"]:
        params = ['learning_rate', 'weight_decay', 'erm_batch_size']
    
    elif args.method == "jtt":
        params = ['learning_rate', 'weight_decay', 'erm_batch_size', 'up', 'T']
        
    elif "mt" in args.method and "l1" in args.method:
        params = ['learning_rate', 'weight_decay', 'erm_batch_size', 'mt_batch_size', 'erm_weight', 'mt_weight', 'reg_weight']
        
    elif "mt" in args.method and "l1" not in args.method:
        params = ['learning_rate', 'weight_decay', 'erm_batch_size', 'mt_batch_size', 'erm_weight', 'mt_weight']
            
    elif "mt" not in args.method and "l1" in args.method:
        params = ['learning_rate', 'weight_decay', 'erm_batch_size', 'mt_batch_size', 'erm_weight', 'reg_weight']
        
    params_str = ""
    for key, value in vars(args).items():
        if key in params:
            params_str += f"{key}_{value}_"
            
    erm_loader, mt_loader, val_loader, test_loader = my_get_loader()
    data_loaders = {
        "tr": erm_loader,
        "va": val_loader,
        "te": test_loader,
    }
    
    for data_type, data_loader in data_loaders.items():
        print(f"{data_type} size: {len(data_loader.dataset)}\t{data_type} dataloader size: {len(data_loader)}")
       
    # Loading the model
    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT,
        "erm_l1": models.ERM,
        "erm_mt": models.ERM_MT,
        "erm_mt_l1": models.ERM_MT,
        "erm_mt2": models.ERM_MT,
        "erm_mt2_l1": models.ERM_MT,
    }[args.method](args, erm_loader)
    
    print("\nModel architecture: \n")
    for n, p in model.network.named_parameters():
        print(n)
    
    # Setting up ckpt and config file names
    append = ""       
    if 'small' in args.data_path:
        append="_small"
    model_dir = os.path.join(args.output_dir, f"{args.dataset}{append}/", f"{args.method}")
    
    model_config_dir = os.path.join(f"{args.output_dir}_params", f"{args.dataset}{append}/", args.method)
    
    model_dir = model_dir.replace("erm_mt_l1", "erm_mt_l2")
    model_config_dir = model_config_dir.replace("erm_mt_l1", "erm_mt_l2")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(model_config_dir):
        os.makedirs(model_config_dir)
        
    best_model_ckpt = os.path.join(model_dir, f"{args.dataset}{append}_{args.method}_{params_str}seed_{args.seed}_best.pt")
    best_model_config_file = os.path.join(model_config_dir, f"{args.dataset}{append}_{args.method}_{params_str}seed_{args.seed}_best.json")
        
    print(f"\nbest model ckpt: {best_model_ckpt}")
    print(f"\nbest model config file: {best_model_config_file}")
    
    best_model_config = dict()
    best_model_config['config'] = {key: val for key, val in vars(args).items()}
    
    # Training loop
    last_epoch = 0
    patience = 0 
    best_selec_val = float('-inf')

    for epoch in range(last_epoch, args.num_train_epochs):
        start_time = time.time()
        result = {
            "epoch": epoch, 
            "time": time.time() - start_time,
        }
        train_loss = 0.0
        total_backward_steps = 0
        total_update_steps = 0
        
        if "mt" in args.method:
            print("\nMT Training...\n\n")
            erm_loss = 0.0
            mt_loss = 0.0
            mt_accum, erm_accum = args.mt_grad_acc, args.grad_acc
            
            if "l1" in args.method:
                l1_loss = 0.0
            
            for index, (i, x, y, g) in enumerate(tqdm(erm_loader, desc="Training Iteration")):
                mt_batch, mt_loader = try_get_mt_batch(mt_loader)
                loss = model.update(index, i, [x, mt_batch], y, g, epoch)
                total_backward_steps += 1
                erm_loss += loss[0]
                mt_loss += loss[1]
                
                if "l1" in args.method:
                    l1_loss += loss[2]
                    train_loss += ((args.erm_weight*loss[0]) + (args.mt_weight*loss[1])  + (args.reg_weight*loss[2]))
                else:
                    train_loss += ((args.erm_weight*loss[0]) + (args.mt_weight*loss[1]))
                
                if (index % erm_accum == 0):
                    if mt_accum > erm_accum: 
                        num_extra_batches = mt_accum - erm_accum
                        for mt_index in range(num_extra_batches):
                            mt_batch, mt_loader = try_get_mt_batch(mt_loader)
                            is_last = mt_index == (num_extra_batches - 1)
                            loss = model.update(-1, i, [None, mt_batch], y, g, epoch, update_params=is_last)
                            total_backward_steps += 1
                            erm_loss += loss[0]
                            mt_loss += loss[1]
                            if "l1" in args.method:
                                l1_loss += loss[2]
                                train_loss += ((args.erm_weight*loss[0]) + (args.mt_weight*loss[1])  + (args.reg_weight*loss[2]))
                            else:
                                train_loss += ((args.erm_weight*loss[0]) + (args.mt_weight*loss[1]))
                            if is_last:
                                total_update_steps += 1
                                is_last=False
                    elif mt_accum == erm_accum:
                        model.step_params()
                        total_update_steps += 1
                    else:
                        raise ValueError('MT accumulation steps should be more than the ERM accumulation steps')
        
        else:
            print("\nNormal Training...\n\n")
            if epoch == args.T + 1 and args.method == "jtt":
                loaders = get_loaders(
                    args.data_path, 
                    args.dataset, 
                    args.erm_batch_size, 
                    args.method,
                    model.weights.tolist()
                )
                erm_loader = loaders["tr"]
                 
            if "l1" in args.method:
                l1_loss = 0.0
                     
            for index, (i, x, y, g) in enumerate(tqdm(erm_loader, desc="Training Iteration")):
                loss = model.update(index, i, x, y, g, epoch)
                total_backward_steps += 1
                total_update_steps += 1
                if "l1" in args.method:
                    l1_loss += loss[1]
                    train_loss += ((args.erm_weight*loss[0]) + (args.reg_weight*loss[1]))
                else:
                    train_loss += loss

        result['tr_loss'] = train_loss/len(erm_loader)
        
        if "mt" in args.method:
            result['erm_loss'] = erm_loss/len(erm_loader)
            result['weighted_erm_loss'] = (args.erm_weight * erm_loss)/len(erm_loader)
            result['mt_loss'] = mt_loss/len(erm_loader)
            result['weighted_mt_loss'] = (args.mt_weight * mt_loss)/len(erm_loader)
        if "l1" in args.method:
            result['l1_loss'] = l1_loss/len(erm_loader)
            result['weighted_l1_loss'] = (args.reg_weight * l1_loss)/len(erm_loader)
            
        result['total_backward_steps'] = total_backward_steps
        result['total_update_steps'] = total_update_steps
    
        preds_gold = dict()
        for loader_name, loader in data_loaders.items():
            if loader_name != 'tr':
                avg_acc, group_accs, inference_loss, preds, gold, group_labels = model.accuracy(loader, loader_name)
                result[f"{loader_name}_loss"] = inference_loss
            else:
                avg_acc, group_accs, preds, gold, group_labels = model.accuracy(loader, loader_name)
                
            group_accs = [round(group_accs, 8) for group_accs in group_accs]
            result["avg_acc_" + loader_name] = avg_acc
            result["group_wise_acc_" + loader_name] = group_accs
            result["std_" + loader_name] = np.std(group_accs)
            result["min_acc_" + loader_name] = min(group_accs)
            result["group_ordering_" + loader_name] = np.argsort(group_accs)

            preds_gold[loader_name] = {
                'preds': preds,
                'gold': gold,
                'group_labels': group_labels
            }

        selec_value = {
            "min_acc_va": result["min_acc_va"],
            "avg_acc_va": result["avg_acc_va"],
        }

        if selec_value['min_acc_va'] > best_selec_val:
            patience = 0
            model.best_selec_val = selec_value['min_acc_va']
            best_selec_val = selec_value['min_acc_va']
            
            print(f"saving best model and results at epoch: {epoch}")
            
            if os.path.exists(best_model_ckpt):
                os.remove(best_model_ckpt)
            
            best_model_config['best_results'] = {
                'epoch': epoch,
                'min_acc_tr': result['min_acc_tr'],
                'avg_acc_tr': result['avg_acc_tr'],
                'group_wise_acc_tr': result['group_wise_acc_tr'],
                'min_acc_va': result['min_acc_va'],
                'avg_acc_va': result['avg_acc_va'],
                'group_wise_acc_va': result['group_wise_acc_va'],
                'min_acc_te': result['min_acc_te'],
                'avg_acc_te': result['avg_acc_te'],
                'group_wise_acc_te': result['group_wise_acc_te'],
                'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            json_object = json.dumps(best_model_config, indent=4)
            with open(best_model_config_file, "w") as outfile:
                outfile.write(json_object)
            outfile.close()
            del json_object
            
            # model.save(best_model_ckpt)
            save_results(preds_gold, args, params_str, epoch)
            
        if "jtt" in args.method:
            if int(args.T)+1 >= epoch:
                patience == 0
            else:
                patience += 1
                print(f"\npatience: {patience}")
                
        else:
            patience += 1
            print(f"\npatience: {patience}")

        print(f"\n\nEpoch {epoch} results: \n")
        for key, value in result.items():
            print(f"{key}:\t{value}")
            
        if patience == 10:
            print("\n\nStopping training as min_acc_va has not increased for 10 consecutive epochs...\n\n")
            break
        
        del result
        gc.collect()
        torch.cuda.empty_cache()
    
        # reset the loaders here
        if args.reset_loader_every_epoch:
            del loader
            del erm_loader
            del mt_loader
            gc.collect()
            torch.cuda.empty_cache()
            erm_loader, mt_loader, _, _ = my_get_loader()



if __name__ == "__main__":
    
    args = parse_args()
    args.pid = os.getpid()
    set_random_seed(args.seed)
    run_experiment(args)