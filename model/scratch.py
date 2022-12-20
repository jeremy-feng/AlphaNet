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