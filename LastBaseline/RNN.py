import torch
import torch.nn as nn
import torch.optim as optim


class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleRNNModel, self).__init__()
        self.linear1 = nn.Linear(input_size, num_classes)
        self.rnn1 = nn.RNN(num_classes, num_classes, batch_first=True)
        self.rnn2 = nn.RNN(num_classes*2, num_classes*2, batch_first=True)
        self.linear2 = nn.Linear(num_classes*4, num_classes)
        self.rnn3 = nn.RNN(num_classes, num_classes, batch_first=True)
        self.rnn4 = nn.RNN(num_classes*2, num_classes*2, batch_first=True)
        self.linear3 = nn.Linear(num_classes*4, num_classes)

    def forward(self, x):
        print(x.shape)
        x = x.permute(0, 2, 1)

        out = self.linear1(x)
        print(out.shape)
        x = out
        out, _ = self.rnn1(out)
        out = torch.cat([x, out], dim=2)
        print(out.shape)
        x = out
        out, _ = self.rnn2(out)
        out = torch.cat([x, out], dim=2)
        print(out.shape)
        out = self.linear2(out)
        print(out.shape)
        x = out
        out, _ = self.rnn3(out)
        out = torch.cat([x, out], dim=2)
        print(out.shape)
        x = out
        out, _ = self.rnn4(out)
        out = torch.cat([x, out], dim=2)
        print(out.shape)
        out = self.linear3(out)
        print(out.shape)
        return out

    def GetParam(self):
        return self.parameters()


class LossFN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L

    def celoss(self, prediction, label):
        fn = torch.nn.CrossEntropyLoss()
        return fn(prediction, label)

    def temporal(self, prediction, label):
        L = []
        T = prediction.shape[2]
        for t in range(T):
            L.append(self.celoss(prediction[:, :, t], label))
        return torch.stack(L).mean()

    def forward(self, nn, x, label):
        if self.args.loss == 'pnnloss':
            return self.standard(nn(x), label)
        elif self.args.loss == 'celoss':
            return self.celoss(nn(x), label)
        elif self.args.loss == 'temporal':
            L = []
            T = x.shape[2]
            for t in range(T):
                L.append(self.celoss(nn(x[:, :, t]), label))
            return torch.stack(L).mean()


# # Hyperparameters
# input_size = 10  # Example input size (number of features)
# hidden_size = 20  # Number of features in the hidden state
# output_size = 1  # Example output size (number of classes)
# num_classes = 5  # Example number of classes

# # Initialize the model, loss function, and optimizer
# model = SimpleRNNModel(input_size, num_classes)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Example input tensor (batch_size, seq_length, input_size)
# # Example batch size of 32 and sequence length of 15
# example_input = torch.randn(32, 15, input_size)

# # Forward pass
# output = model(example_input)
# print(output.shape)  # Should output a tensor with shape (32, 15, output_size)
