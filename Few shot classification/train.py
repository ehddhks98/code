import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from backbone import get_backbone
from model import get_model
from utils import extract_data, remap_labels
from dataset import episode_dataloader, get_dataset, normal_dataloader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="/home/work/study/dongwan/MatchingNet/runs/1shot")


def train(model, dataloader, optimizer, n_epoch, model_name):
    model.train()
    for epoch in range(n_epoch):
        total_loss = 0.0
        correct = 0
        total = 0

        if model_name in ['baseline', 'baseline_pp']:
            # 각 에포크에서 여러 배치에 대해 반복
            for batch_idx, batch in enumerate(dataloader):
                data, label = batch

                # 데이터를 GPU로 이동 (가능한 경우)
                data, label = (data.to(device), label.to(device))

                prob = model(data, label)

                loss = F.nll_loss(prob, label)            

                # 정확도 계산
                label_prediction = prob.argmax(dim=1)
                correct += (label_prediction == label).sum().item()
                total += label.size(0)

                # 역전파 및 옵티마이저 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 배치당 손실 누적
                total_loss += loss.item()

                # 현재 배치 진행 상태 출력 (옵션)
                print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

                # 텐서보드에 배치당 손실 기록
                writer.add_scalar(
                    "Loss/batch/train", loss.item(), epoch * len(dataloader) + batch_idx
                )

            # 에포크당 평균 손실 및 정확도 출력
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            print(f"Epoch [{epoch+1}/{n_epoch}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # 텐서보드에 기록
            writer.add_scalar("Loss/epoch/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/epoch/train", accuracy, epoch)

            writer.close()

        else:
            # 각 에포크에서 여러 배치에 대해 반복
            for batch_idx, (support_set, query_set) in enumerate(dataloader):
                # support_set과 query_set에서 데이터와 레이블을 추출
                support_data, support_label, query_data, query_label = extract_data(support_set, query_set)
                support_label, query_label = remap_labels(support_label, query_label)

                # 데이터를 GPU로 이동 (가능한 경우)
                support_data, query_data, support_label, query_label = (
                    support_data.to(device),
                    query_data.to(device),
                    support_label.to(device),
                    query_label.to(device),
                )

                prob = model(support_data, query_data, support_label)

                # 손실 계산 (소프트맥스 함수 사용)
                prob = F.softmax(prob)
                loss = F.nll_loss(prob, query_label)            

                # 정확도 계산
                label_prediction = prob.argmax(dim=1)
                correct += (label_prediction == query_label).sum().item()
                total += query_label.size(0)

                # 역전파 및 옵티마이저 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 배치당 손실 누적
                total_loss += loss.item()

                # 현재 배치 진행 상태 출력 (옵션)
                print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

                # 텐서보드에 배치당 손실 기록
                writer.add_scalar(
                    "Loss/batch/train", loss.item(), epoch * len(dataloader) + batch_idx
                )

            # 에포크당 평균 손실 및 정확도 출력
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            print(f"Epoch [{epoch+1}/{n_epoch}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # 텐서보드에 기록
            writer.add_scalar("Loss/epoch/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/epoch/train", accuracy, epoch)

            writer.close()
            
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--model_name', default = 'baseline', type=str, required=True)
    parser.add_argument('--learning_rate', default = '0.001', type=float)
    parser.add_argument('--n_epoch', default = '10', type=int, required=True)
    parser.add_argument('--n_support', default = '10', type=int, required=True)
    args = parser.parse_args()

    train_set, test_set = get_dataset()
    train_loader_normal = normal_dataloader(train_set, args.n_batch_size, args.n_epoch)
    train_loader_episode = episode_dataloader(train_set, args.n_class, args.n_support, args.n_query, args.n_batch, args.n_epoch)
    test_loader_episoe = episode_dataloader(test_set, args.n_class, args.n_support, args.n_query, args.n_batch, args.n_epoch)

    model = get_model(args.model_name)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 모델 학습
    model = train(model, train_loader_normal)


    # 모델 저장
    torch.save(model.state_dict(), f"{args.model}_initial_trained_model.pth")


