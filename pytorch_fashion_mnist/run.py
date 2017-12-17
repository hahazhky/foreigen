import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from fashionmnist import FashionMNIST
from net import sim_fc


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])
f_mnist = {'train': FashionMNIST('./fashion/', train=True, download=False, transform=train_transform),
           'test': FashionMNIST('./fashion/', train=False, download=False, transform=test_transform)}

dataloaders = {'train': torch.utils.data.DataLoader(f_mnist['train'], batch_size=128, shuffle=True, num_workers=4),
               'test': torch.utils.data.DataLoader(f_mnist['test'], batch_size=128, shuffle=False, num_workers=4)}

data_sizes = {x: len(f_mnist[x]) for x in ['train', 'test']}

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch:{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # print(outputs.size())
                _, preds = torch.max(outputs.data, 1)
                # print(preds)
                loss = criterion(outputs, labels)

                # output and optimize if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                  phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


model = sim_fc()
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

optimizer_ft = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler = None

model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(model_ft, 'model.pt')
