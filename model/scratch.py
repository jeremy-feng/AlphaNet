import torch

# if test_loss < best_test_loss:
    #     torch.save(alphanet, f'./BestModels/Best_{seed}_alphanet_model_{today}.pth')
    #     best_test_loss = test_loss
    #     best_test_epoch = epoch
    # if epoch - best_test_epoch > 10:
    #     print(f'{seed}_best_test_loss', best_test_loss)
    #     print(f'{seed}_best_test_epoch', best_test_epoch)
    #     break

torch.save(alphanet, './model_v1_pool.pth')


net = torch.load("./model_v1_pool.pth")

net.eval()

pred = net(data).detach().numpy()

true = label.detach().numpy()

plt.plot(pred[:100], label='pred')
plt.plot(true[:100], label='true')
plt.legend()
plt.show()


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch.float), target.to(torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss  # sum up batch loss
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\n Train Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch.float), target.to(torch.float)
            output = model(data)
            pred = output
            test_loss += criterion(pred, target) # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


model = alphanet
for epoch in range(1, 10 + 1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)