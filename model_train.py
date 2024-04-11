import torch
import torchvision
from pathlib import Path

# https://stackoverflow.com/questions/918154/relative-paths-in-python
#using pathlib instead of os library

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class mnist():
    def __init__(self):
        self.n_epochs = 3
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.random_seed)
        self.train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(str(Path(__file__).parent), train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=self.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(str(Path(__file__).parent), train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=self.batch_size_test, shuffle=True)
        self.examples = enumerate(self.test_loader)
        self.batch_idx, (self.example_data, self.example_targets) = next(self.examples)

        self.network = Net()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate,
                      momentum=self.momentum)
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]
    
    def train(self, epoch):
        self.create_folder_if_not_exists(str(Path(__file__).parent)+'/model')
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))

                torch.save(self.network.state_dict(), str(Path(__file__).parent)+'/model/model.pth')
                torch.save(self.optimizer.state_dict(), str(Path(__file__).parent)+'/model/optimizer.pth')

    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(self.test_loader.dataset)
            self.test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))

    def create_folder_if_not_exists(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created successfully.")
        else:
            print(f"Folder '{folder_name}' already exists.")
    
    def mainProcess(self):
        self.test()
        for epoch in range(1, self.n_epochs + 1):
            self.train(epoch)
            self.test()

if __name__ == "__main__":
    process = mnist()
    process.mainProcess()
