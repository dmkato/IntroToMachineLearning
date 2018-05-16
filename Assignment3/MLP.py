import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
# import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = getattr(F, activation_func)(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x))

def train(epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        if batch_idx == 4:
            return

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_len = (len(train_loader.dataset) - (32 * 4))
    val_loss /= val_len
    loss_vector.append(val_loss)

    accuracy = 100. * correct / val_len
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, val_len, accuracy))
#
# def validate(loss_vector, accuracy_vector):
#     model.eval()
#     val_loss, correct = 0, 0
#     for data, target in validation_loader:
#         if cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         val_loss += F.nll_loss(output, target).data[0]
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()
#
#     val_len = len(validation_loader.dataset)
#     val_loss /= val_len
#     loss_vector.append(val_loss)
#
#     accuracy = 100. * correct / val_len
#     accuracy_vector.append(accuracy)
#
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         val_loss, correct, val_len, accuracy))

def test():
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_len = len(validation_loader.dataset)
    val_loss /= val_len

    accuracy = 100. * correct / val_len
    print('Results:')
    print('lr = {}'.format(lr))
    print('epochs = {}'.format(epochs))
    print('activation_func = {}'.format(activation_func))
    print('momentum = {}'.format(momentum))
    print('dropout = {}'.format(dropout))
    print('weight_decay = {}'.format(weight_decay))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, val_len, accuracy))

# def show_examples():
#     for (X_train, y_train) in train_loader:
#         print('X_train:', X_train.size(), 'type:', X_train.type())
#         print('y_train:', y_train.size(), 'type:', y_train.type())
#         break
#     for i in range(10):
#         plt.subplot(1,10,i+1)
#         plt.axis('off')
#         image = X_train[i,:,:,:].numpy()
#         plt.imshow(image.reshape(3,32,32).transpose(1,2,0))
#         plt.title('Class: '+str(y_train[i]))

# def plot_results():
#     plt.figure(figsize=(5,3))
#     plt.plot(np.arange(1,epochs+1), lossv)
#     plt.title('Validation Loss {} (Learning Rate = {})'.format(activation_func.title(), lr))
#
#     plt.figure(figsize=(5,3))
#     plt.plot(np.arange(1,epochs+1), accv)
#     plt.title('Validation Accuracy {} (Learning Rate = {})'.format(activation_func.title(), lr))
#     plt.show()

def create_train_loader():
    loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor()
                ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return loader

def create_validation_loader():
    loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor()
                ])),
        batch_size=batch_size, shuffle=False, **kwargs)
    return loader

if __name__ == '__main__':
    lr = 0.0001
    epochs = 1
    activation_func = 'relu'
    momentum = 0.8
    dropout = 0.4
    weight_decay = 0.0001

    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    batch_size = 32
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = create_train_loader()
    validation_loader = create_validation_loader()
    # pltsize=1
    # plt.figure(figsize=(10*pltsize, pltsize))
    # show_examples()
    model = Net()
    if cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    print(model)
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)
    test()
    # plot_results()
