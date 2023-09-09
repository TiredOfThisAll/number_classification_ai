import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Проверяем доступность GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    # Загрузка и предобработка данных
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader

# Определение архитектуры нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 входной канал, 32 выходных канала, ядро 3x3
        self.pool = nn.MaxPool2d(2, 2)   # Максимальный пуллинг 2x2
        self.conv2 = nn.Conv2d(32, 64, 3) # 32 входных канала, 64 выходных канала, ядро 3x3
        self.fc1 = nn.Linear(64 * 5 * 5, 128) # Полносвязный слой с 128 нейронами
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)        # Выходной слой с 10 нейронами (классами)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  # Исправлено на правильный размер
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net():
    trainloader, testloader = load_data()

    # Перемещение модели на GPU/CPU
    net = Net().to(device)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Обучение нейронной сети
    for epoch in range(5):  # количество эпох
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Перемещение данных на устройство
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # печать каждые 100 мини-батчей
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Обучение завершено')

    # Тестирование нейронной сети
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность сети на тестовых данных: {100 * correct / total:.2f}%')

    torch.save(net.state_dict(), 'mnist_model.pth2')

    torch.cuda.empty_cache()
