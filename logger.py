import csv
import uuid
import torch
import os
from pathlib import Path

class Logger:

    def __init__(self, model_arch, model_name, dropout, lr, batch, hidden, arguments, dataset_name):
        self.dataset_name = dataset_name
        self.model_arch = model_arch
        self.model_name = model_name
        self.dropout = dropout
        self.lr = lr
        self.batch = batch
        self.hidden = hidden
        self.id = uuid.uuid4().hex
        if arguments:
            self.arguments = 'arguments'
        else:
            self.arguments = 'facts'
        self.path = "results/" + dataset_name + "/" + model_arch + "/" + model_name + "/" + self.arguments + "/" + self.id + "/"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.ids = None

    def save_model(self, model):
        torch.save(model, self.path + "model.pt")

    def save_dual(self, model1, model2):
        torch.save(model1, self.path + "model1.pt")
        torch.save(model2, self.path + "model2.pt")

    def load_dual(self):
        model1 = torch.load(self.path + "model1.pt")
        model2 = torch.load(self.path + "model2.pt")
        return model1, model2

    def load_model(self):
        model = torch.load(self.path + "model.pt")
        return model

    def save_results(self, results, outputs):

        with open(self.path + "outputs.csv", 'w') as loss_file:
            writer = csv.writer(loss_file)
            for pred, truth, id in zip(outputs[0], outputs[1], self.ids):
                writer.writerow([id, pred, truth])

        csv_columns = ["model_name", "model_architecture", "dropout", "lr", "batch_size",
                       "n_hidden", "val_loss", "val_f1", "val_precision", "val_recall", "val_accuracy",
                       "test_loss", "test_f1", "test_precision", "test_recall", "test_accuracy"]

        new_dict_data = [
            {"model_name": self.model_name, "model_architecture": self.model_arch, "dropout": self.dropout,
             "lr": self.lr, "batch_size": self.batch,
             "n_hidden": self.hidden, "val_loss": results[0], "val_f1": results[1], "val_precision": results[2],
             "val_recall": results[3], "val_accuracy": results[4], "test_loss": results[5], "test_f1": results[6],
             "test_precision": results[7], "test_recall": results[8], "test_accuracy": results[9]}
        ]

        csv_file = self.path + "results.csv"

        if os.path.isfile(csv_file):
            try:
                with open(csv_file, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    for data in new_dict_data:
                        writer.writerow(data)

            except IOError:
                print("I/O error")

        else:
            try:
                with open(csv_file, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in new_dict_data:
                        writer.writerow(data)

            except IOError:
                print("I/O error")
