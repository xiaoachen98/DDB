import numpy as np
import torch
import torch.nn.functional as F


class ClassFeatures:
    def __init__(self, numbers=19, proto_momentum=0.9999, dev=torch.device("cpu")):
        self.class_numbers = numbers
        self.class_features = [[] for _ in range(self.class_numbers)]
        self.dev = dev
        self.num = np.zeros(numbers)
        self.proto_momentum = proto_momentum
        self.objective_vectors_num = torch.zeros([self.class_numbers]).to(self.dev)
        self.objective_vectors = torch.zeros([self.class_numbers, 256]).to(self.dev)

    def calculate_mean_vector(self, feat_cls, outputs):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())

        outputs_pred = outputs_argmax

        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def process_label(self, label):
        batch, _, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(self.dev)
        idx = torch.where(
            label < self.class_numbers,
            label,
            torch.Tensor([self.class_numbers]).to(self.dev),
        )
        pred1 = pred1.scatter_(1, idx.long(), 1)
        return pred1

    def update_objective_SingleVector(
        self, idx, vector, name="moving_average", start_mean=True
    ):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[idx].item() < 100:
            name = "mean"
        if name == "moving_average":
            self.objective_vectors[idx] = (
                self.objective_vectors[idx] * self.proto_momentum
                + (1 - self.proto_momentum) * vector.squeeze()
            )
            self.objective_vectors_num[idx] += 1
            self.objective_vectors_num[idx] = min(self.objective_vectors_num[idx], 3000)
        elif name == "mean":
            self.objective_vectors[idx] = (
                self.objective_vectors[idx] * self.objective_vectors_num[idx]
                + vector.squeeze()
            )
            self.objective_vectors_num[idx] += 1
            self.objective_vectors[idx] = (
                self.objective_vectors[idx] / self.objective_vectors_num[idx]
            )
            self.objective_vectors_num[idx] = min(self.objective_vectors_num[idx], 3000)
            pass
        else:
            raise NotImplementedError(
                "no such updating way of objective vectors {}".format(name)
            )
