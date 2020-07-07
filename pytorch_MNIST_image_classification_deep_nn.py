from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import torch.nn as nn

def accuracy(outputs, labels):
    probs = F.softmax(outputs, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    total_accurate_preds = torch.sum(preds == labels).item()
    total_preds = len(preds)
    return torch.tensor(total_accurate_preds / total_preds)

class MnistModel(nn.Module):
    # Feed forward neural network with 1 hidden layer
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
    def forward(self, xb):
        # Flatten the image tensor
        xb = xb.view(xb.size(0), -1) # 784, -1
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply the activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_acc': acc, 'val_loss': loss}
    def validation_epoch_end(self, metrics):
        batch_acc = [x['val_acc'] for x in metrics]
        epoch_acc = torch.stack(batch_acc).float().mean() # Convert long to float is batch_acc is a long
        batch_loss = [x['val_loss'] for x in metrics]
        epoch_loss = torch.stack(batch_loss).float().mean() # Convert long to float is batch_loss is a long
        return {'val_acc': epoch_acc.item(), 'val_loss': epoch_loss.item()}
    def epoch_end(self, epoch, results):
        print(f"EPOCH [{epoch}]: LOSS, {results['val_loss']}; ACCURACY, {results['val_acc']}")
        
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break

        
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
        
        

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def predict_image(image, model):
    image = image.unsqueeze(0)
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

if __name__ == '__main__':

    directory = r'D:\01_automation\02_offline\ImageClassification\00_image_datasets\torchvision_datasets'
    dataset = MNIST(root=directory, train=True, transform=ToTensor())
    train_ds, valid_ds = random_split(dataset, [50000, 10000])
    batch_size = 128

    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

#     show_batch(train_dl)
#     show_batch(valid_dl)
    input_size = 784
    hidden_size = 64
    num_classes = 10

    model = MnistModel(in_size=input_size, hidden_size=hidden_size, out_size=num_classes)
    history = [evaluate(model, valid_dl)]
    history += fit(5, 0.5, model, train_dl, valid_dl)
    history += fit(5, 0.1, model, train_dl, valid_dl)
    
    # SAVE THE MODEL TO A DIRECTORY    
    trained_model= r'D:\01_automation\02_offline\ImageClassification\00_image_datasets\torchvision_datasets\MNIST\trained_model'
    torch.save(model.state_dict(), os.path.join(trained_model, 'mnist_deep_nn_2020-07-07.pth'))
    
    model2 = MnistModel(in_size=input_size, hidden_size=hidden_size, out_size=num_classes)
    model2.load_state_dict(torch.load(os.path.join(trained_model, 'mnist_deep_nn_2020-07-07.pth')))
    test_dataset = MNIST(root=directory, train=False, transform=ToTensor())
    img, label = test_dataset[353]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model2))
