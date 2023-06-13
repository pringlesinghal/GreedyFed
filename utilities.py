import torch


def accuracy_test(scores, labels):
    _, predictions = scores.max(1)
    num_correct = torch.sum(predictions == labels)
    total = len(labels)
    accuracy = num_correct / total
    return accuracy


def model_accuracy(serverModel, test_data, test_targets, criterion, device):
    with torch.no_grad():
        scores = serverModel(test_data)
        loss = criterion(scores, test_targets)
        accuracy = accuracy_test(scores, test_targets)
    return loss, accuracy
