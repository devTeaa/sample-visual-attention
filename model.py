import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.fc1a = nn.Linear(16 * 5 * 5, 120)
        self.fc2a = nn.Linear(120, 84)
        self.fc3a = nn.Linear(84, 25)

    def forward(self, x):
        x = self.conv1(x)
        # 6 x 28 x 28
        x = F.relu(x)
        # 6 x 28 x 28
        x = self.pool(x)
        # 6 x 14 x 14
        # x = self.pool(F.relu(self.conv1(x)))

        x = self.conv2(x)
        # 16 x 10 x 10
        x = F.relu(x)
        # 16 x 10 x 10
        x = self.pool(x)
        # 16 x 5 x 5
        # x = self.pool(F.relu(self.conv2(x)))

        # set to 1 dimension
        y = x.view(-1, 16 * 5 * 5)
        y = F.relu(self.fc1a(y))
        y = F.relu(self.fc2a(y))
        y = self.fc3a(y)
        # get the first layer of y and set it back 5 x 5
        y = y[0].view(-1, 5)

        z = x * y
        print(x.size())
        print(y.size())
        print(z.size())
        print("layer x[0][0]", x[0][0])
        print("layer y", y)
        print("x[0][0] * y[0][0]", z[0][0])

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for i, data in enumerate(trainloader, 0):
    inputs, labels = data

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break

print('Finished training')
