import json
import torch


if __name__ == '__main__':
    with open('../data/vocab.txt', 'r') as f:
        vocab = json.load(f)
        