#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("-pb", "--progress_bar", help="Use  progress bar for training", type=bool)
args = parser.parse_args()

import rl4fisheries


from rl4fisheries.utils import sb3_train    
sb3_train(args.file)
