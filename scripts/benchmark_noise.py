#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("-pb", "--progress_bar", help="Use  progress bar for training", type=bool)
parser.add_argument("-noise", "--noise", help="sigma for the noise level to use", type=float)
args = parser.parse_args()

import rl4fisheries


from rl4fisheries.utils import sb3_train    
sb3_train(args.file, progress_bar = args.progress_bar, config={"sigma": args.noise})
