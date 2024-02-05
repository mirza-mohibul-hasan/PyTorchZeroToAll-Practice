from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


# The code defines a class named Model, which is a subclass of nn.Module. This is a common practice in PyTorch when creating neural network models.
class Model(nn.Module):
    def __init__(self):
        # super(Model, self).__init__(): This line calls the constructor of the parent class (nn.Module). It's necessary to include this line in the constructor of your custom model class.
        super(Model, self).__init__()

        # self.linear = torch.nn.Linear(1, 1): This line creates an instance of the nn.Linear module. The nn.Linear module represents a linear transformation, i.e., a linear layer in a neural network. It takes two arguments: the number of input features (1 in this case) and the number of output features (1 in this case). In the context of a simple linear regression model, this linear layer corresponds to the weight and bias terms in the regression equation.
        self.linear = torch.nn.Linear(1, 1)

    # The forward method defines the forward pass of the model. It takes an input tensor x and returns the predicted output tensor. In this case, it applies the linear layer (self.linear) to the input tensor:
    # y_pred = self.linear(x): This line computes the forward pass by applying the linear transformation to the input tensor. The result (y_pred) represents the predicted output of the model.
    def forward(self, x):
        y_pred = self.linear(x)

        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Aftter Training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())
