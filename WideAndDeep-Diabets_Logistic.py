from torch import nn, optim, from_numpy
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)

x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(8, 6)
        self.layer2 = nn.Linear(6, 4)
        self.layer3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output1 = self.sigmoid(self.layer1(x))
        output2 = self.sigmoid(self.layer2(output1))
        y_pred = self.sigmoid(self.layer3(output2))
        return y_pred


# our model
model = Model()


loss_function = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(10000):
    y_pred = model(x_data)

    loss = loss_function(y_pred, y_data)

    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
