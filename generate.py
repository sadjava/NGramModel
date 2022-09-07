import argparse
import pickle
import re
from train import NGramModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--prefix', type=str)
parser.add_argument('--length', type=int)
args = parser.parse_args()

args.prefix = "Стоит заметить, что"
args.length = 25
args.model = "model.pkl"

if args.prefix:
    args.prefix = re.sub("\W+", " ", args.prefix.lower())

with open(args.model, "rb") as file:
    model = pickle.load(file)

generated = model.generate(args.prefix, args.length)
print(generated)
