# COMPUTE THE ACCURACY OF PREDICTIONS AGAINST LABELS
def accuracy(outputs, labels):
    probs = F.softmax(outputs, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    total_correct_predictions = torch.sum(preds==labels).item()
    total_preds = len(preds)
    return torch.tensor(total_correct_predictions/total_preds)

# PSEUDO-CODE FOR THE TRAINING AND VALIDATION OF DATASETS
# for epoch in range(epoch_num):
#     # TRAINING THE MODEL
#     for images, labels in train_dl:
#         # GENERATE PREDICTIONS
#         # CALCULATE LOSS
#         # COMPUTE GRADIENTS
#         # UPDATE WEIGHTS WITH RESPECT TO GRADIENTS
#         # RESET GRADIENTS TO ZERO
#     # VALIDATE THE MODEL
#     for images, labels in val_dl:
#         # GENERATE PREDICTIONS
#         # CALCULATE LOSS
#         # CALCULATE METRICS (INCLUDING ACCURACY)    
#     # CALCULATE AVERAGE VALIDATION LOSS AND METRICS
#     # LOG EPOCH, LOSS, AND METRICS FOR INSPECTION
    
    
class MnistModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_SIZE, NUM_CLASSES)
    
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # GENERATE PREDICTIONS
        loss = F.cross_entropy(out, labels) # COMPUTE LOSS
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # GENERATE PREDICTIONS
        loss = F.cross_entropy(out, labels)   # COMPUTE LOSS
        acc = accuracy(out, labels)           # COMPUTE ACCURACY
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # COMBINE LOSSES
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_fn=torch.optim.SGD):
    history = []
    optimizer = opt_fn(model.parameters(),lr)
    for epoch in range(epochs):
        # TRAINING PHASE
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


if __name__ == '__main__':
    training_complete=True
    if training_complete:
        pass
    else:
        # DIRECTORY TO STORE IMAGES
        directory = r'D:\01_automation\02_offline\ImageClassification\00_image_datasets\torchvision_datasets'
        # DOWNLOAD DATASET AND CONVERT IMAGES TO PYTORCH TENSORS
        from torchvision.datasets import MNIST
        import torchvision.transforms as transforms
        dataset = MNIST(root=directory, download=True, train=True, transform=transforms.ToTensor())
        # SPLIT THE DATASET INTO TRAINING DATASET AND VALIDATION DATASET
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [50000, 10000])
        # len(train_ds), len(val_ds)
        # CONVERT DATASETS TO DATA LOADERS WITH A SPECIFIED BATCH SIZE
        BATCH_SIZE = 128
        from torch.utils.data import DataLoader
        train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, BATCH_SIZE) # NO NEED TO SHUFFLE VALIDATION DATA LOADER
        model = MnistModel()
        result0 = evaluate(model, val_loader)
        history1 = fit(5, 0.001, model, train_loader, val_loader)
        history2 = fit(5, 0.001, model, train_loader, val_loader)
        history3 = fit(5, 0.001, model, train_loader, val_loader)
        history4 = fit(5, 0.001, model, train_loader, val_loader)
        history5 = fit(5, 0.001, model, train_loader, val_loader)
        history6 = fit(20, 0.001, model, train_loader, val_loader)

        
# #     DISPLAY THE CURRENT ACCURACY OF THE MODEL
#     test_loader = DataLoader(test_dataset, batch_size=256)
#     result = evaluate(model, test_loader)
#     print(result)

#     # SAVE THE MODEL TO A DIRECTORY
#     import os
#     trained_model= r'D:\01_automation\02_offline\ImageClassification\00_image_datasets\torchvision_datasets\MNIST\trained_model'
#     torch.save(model.state_dict(), os.path.join(trained_model, 'mnist-logistic.pth'))
    
    
    model2 = MnistModel()
    model2.load_state_dict(torch.load(os.path.join(trained_model, 'mnist-logistic.pth')))
#     model2.state_dict()
    test_dataset = MNIST(root=directory, train=False, transform=transforms.ToTensor())
    img, label = test_dataset[353]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model2))
