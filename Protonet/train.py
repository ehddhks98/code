import torch
from torch import nn, optim
from tqdm import tqdm
from easyfsl.utils import sliding_average

def fit(model, optimizer, criterion, support_images, support_labels, query_images, query_labels):
    optimizer.zero_grad()
    classification_scores = model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, train_loader, optimizer, criterion, log_update_frequency=10):
    all_loss = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
            loss_value = fit(model, optimizer, criterion, support_images, support_labels, query_images, query_labels)
            all_loss.append(loss_value)
            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))
