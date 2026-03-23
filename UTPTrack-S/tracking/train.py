import os
import argparse


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple", help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--use_wandb', type=int, default=0, choices=[0, 1])  # whether to use wandb
    
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d --use_wandb %d" \
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.use_wandb)
    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --use_wandb %d" \
                    % (args.nproc_per_node, args.script, args.config, args.save_dir, args.use_lmdb, args.use_wandb)
    elif args.mode == "multi_node":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_addr %s --master_port %d --nnodes %d --node_rank %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --use_wandb %d " \
                    % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank, args.script, args.config, args.save_dir, args.use_lmdb, args.use_wandb)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    print(">>>>>>>>>>>>>>>>>>>>> train cmd >>>>>>>>>>>>>>>>>>>>>")
    print(train_cmd)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
