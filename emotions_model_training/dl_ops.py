import mlflow.types
import torch
import mlflow

from mlflow.models import infer_signature

class dl_ops():
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.run_iter = 0

    def train(self, dataloader, device):
        # Setting proper mode
        self.model.train()

        signature = None
        input_example = None

        for iteration, batch in enumerate(dataloader):
            sample = batch['image'].to(device)
            label = batch['label'].to(device)

            pred = self.model(sample)
            loss = self.loss_fn(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            loss_item, current = loss.item(), (iteration + 1) * len(sample)
            print(f"Batch: {iteration + 1} ({current / len(dataloader.dataset)}) \n Loss: {loss_item:>3f}")

            self.run_iter += 1
            mlflow.log_metric("train_loss", loss_item, self.run_iter)

    def valid(self, dataloader, device, epoch):
        self.model.eval()

        correct_pred=0
        accuracy = 0
        valid_loss = 0

        for iteration, batch in enumerate(dataloader):
            sample = batch['image'].to(device)
            label = batch['label'].to(device)

            pred = self.model(sample)
            valid_loss = self.loss_fn(pred, label)
            
            correct_pred += (pred.argmax(1) == label).type(torch.float).sum().item()
            
            current = (iteration + 1) * len(sample)
            print(f"Batch: {iteration + 1} ({current / len(dataloader.dataset)})")

        valid_loss /= len(dataloader)
        accuracy = correct_pred / len(dataloader.dataset)
        accuracy *= 100

        mlflow.log_metric("accuracy_val", accuracy, epoch)
        mlflow.log_metric("loss_val", valid_loss, epoch)
        mlflow.log_metric("correct_pred", correct_pred, epoch)

        print(f"Validation Loss: {valid_loss:>3f} % \n Accuracy of validation: {accuracy:>3f}")

    def test(self):
        self.model.test()