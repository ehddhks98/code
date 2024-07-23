from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random


from dataset import OmniglotDataset
from dataset import get_dataloaders
from embedding import CNNEncoder
from module import PrototypicalNetwork
from train import train_protonet
from test import test_protonet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    
    encoder = CNNEncoder().to(device)
    model = PrototypicalNetwork(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_protonet(model, train_loader, optimizer, args.n_way_train, args.n_support, args.n_query_train, device)
    acc = test_protonet(model, test_loader, args.n_way_test, args.n_support, args.n_query_test, device)
   
    
    print(f"Epoch {epoch+1}, Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototypical Networks for Few-shot Learning on Omniglot")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to Omniglot dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--n_way_train", type=int, default=60, help="Number of classes in training episodes")
    parser.add_argument("--n_way_test", type=int, default=5, help="Number of classes in test episodes")
    parser.add_argument("--n_support", type=int, default=1, help="Number of support samples per class")
    parser.add_argument("--n_query_train", type=int, default=5, help="Number of query samples per class in training")
    parser.add_argument("--n_query_test", type=int, default=5, help="Number of query samples per class in testing")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")

    args = parser.parse_args()
    main(args)