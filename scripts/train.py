#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("-pb", "--progress_bar", help="Use  progress bar for training", type=bool)
parser.add_argument("-rppo", "--recurrent-ppo", help="Hyperpar structure for recurrent ppo.", type=bool, default=False)
args = parser.parse_args()

import rl4fisheries

if args.recurrent_ppo:
    from rl4fisheries.utils import sb3_train_v2
    sb3_train_v2(args.file, progress_bar = args.progress_bar)
else:
    from rl4fisheries.utils import sb3_train
    sb3_train(args.file, progress_bar = args.progress_bar)