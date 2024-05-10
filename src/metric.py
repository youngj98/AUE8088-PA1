from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('true_positives', default = torch.tensor(0), dist_reduce_fx = 'sum')
        self.add_state('false_positives', default = torch.tensor(0), dist_reduce_fx = 'sum')
        self.add_state('false_negatives', default = torch.tensor(0), dist_reduce_fx = 'sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim = 1)
        assert preds.shape == target.shape, "Not equal"

        true_positives = torch.sum((preds == 1) & (target == 1))
        false_positives = torch.sum((preds == 1) & (target == 0))
        false_negatives = torch.sum((preds == 0) & (target == 1))

        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives


    def compute(self):
        if (self.true_positives + self.false_positives) > 0:
            precision = self.true_positives.float() / (self.true_positives + self.false_positives).float()
        else:
            precision = 0

        if (self.true_positives + self.false_negatives) > 0:
            recall = self.true_positives.float() / (self.true_positives + self.false_negatives).float()
        else:
            recall = 0

        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        return f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim = 1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Not equal"

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
