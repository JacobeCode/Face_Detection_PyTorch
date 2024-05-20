class dl_ops():
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    def train(self, dataloader, epoch, device):
        # Setting proper mode
        self.model.train()

        for iteration, batch in enumerate(dataloader):
            sample = batch['image'].to(device)
            label = batch['label'].to(device)

            pred = self.model(sample)
            loss = self.loss_fn(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if iteration % 50 == 0:
            loss_item, current = loss.item(), (iteration + 1) * len(sample)
            print(f"Batch: {iteration + 1} ({current / len(dataloader.dataset)}) \n Loss: {loss_item:>3f}")

    def valid():
        pass
    def test():
        pass