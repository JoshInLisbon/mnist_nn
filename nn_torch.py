import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()]) # divides by 255.

X = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
y = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

bs = 64
train_loader = torch.utils.data.DataLoader(X, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(y, batch_size=bs)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.input_to_hidden = nn.Linear(784, 10)
    self.relu = nn.ReLU()
    self.hidden_to_output = nn.Linear(10, 10)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    # x => (batch_size, 28, 28)
    x = x.view(-1, 784) # (bs, 784)
    z1 = self.input_to_hidden(x) # (bs, 10)
    a1 = self.relu(z1) # (bs, 10)
    z2 = self.hidden_to_output(a1) # (bs, 10)
    a2 = self.softmax(z2) # (bs, 10)

    return a2

model = NeuralNetwork()
lossfn = nn.CrossEntropyLoss()
# stochastic gradient decent
learning_rate = 0.1
optimiser = optim.SGD(model.parameters(), lr=learning_rate)

def evaluate_model(model, data_loader, in_train_loop=False):
  model.eval() # eval mode
  correct, total = 0, 0

  with torch.no_grad():
    for X, y in data_loader:
      outputs = model(X)
      _, predictions = torch.max(outputs, 1)
      total += y.size(0)
      correct += (predictions == y).sum().item()

  if in_train_loop:
    model.train()

  return correct / total

def train_model(model, train_loader, lossfn, optimiser, epochs):
  model.train() # training mode
  for epoch in range(1, epochs + 1):
    for i, (X_train, y_train) in enumerate(train_loader):
      # forward:
      outputs = model(X_train)
      loss = lossfn(outputs, y_train)

      # backward propagation:
      optimiser.zero_grad() # clear prev gradients
      loss.backward() # find grads from loss
      optimiser.step() # SGD

    if epoch % 5 == 0:
      accuracy = evaluate_model(model, train_loader, in_train_loop=True)
      print(f"Epoch {epoch}")
      print(f"Loss: {loss.item():.4f}")
      print(f"Accuracy: {accuracy:.4f}\n")

epochs = 100
train_model(model, train_loader, lossfn, optimiser, epochs)

accuracy = evaluate_model(model, test_loader)
print(f"Test accuracy: {accuracy:.4f}\n")