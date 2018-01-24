import numpy
import torch
import dilated_rnn
import os

V = torch.autograd.Variable


def toydata(T, n):
    x = numpy.zeros((T + 20, n))

    x[:10] = numpy.random.randint(0, 8, size=[10, n])
    x[10:T + 9] = 8
    x[T + 9:] = 9
    return torch.from_numpy(x.astype(int))


class Copy(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = torch.nn.Embedding(10, 10)

        self.drnn = dilated_rnn.DilatedRNN(
            torch.nn.GRU,
            10,
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
            [10] * 9,
            0.0
        )

        self.project = torch.nn.Linear(10, 8)

    def forward(self, input):
        return self.drnn(self.embed(input))[-10:]


model = Copy()
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
criterion = torch.nn.CrossEntropyLoss()

os.system('echo "Iteration,Cross-Entropy" > log.csv')

it = 0
while True:

    batch = toydata(20, 100)

    optimizer.zero_grad()

    output = model(V(batch))

    loss = 0
    for j in range(output.size(1)):
        loss += criterion(output[:, j, :], V(batch[:10, j]))
    loss = loss / output.size(1)

    loss.backward()
    optimizer.step()

    if it % 10 == 0:
        print(loss.data[0])

    it += 1

    os.system("echo '{},{}' >> log.csv".format(it, loss.data[0]))