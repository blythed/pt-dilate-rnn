import unittest
import torch
from dilated_rnn import DilatedRNN


use_cuda = torch.cuda.is_available()


class TestStackInputs(unittest.TestCase):
    def test(self):
        
        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.randn(15, 2, 13)

        if use_cuda:
            x = x.cuda()
            drnn.cuda()

        for rate in [2, 8]:

            padded = drnn._padinputs(x, rate)

            self.assertEqual(padded.size(0), 16)

        for rate in [3, 5]:

            padded = drnn._padinputs(x, rate)

            self.assertEqual(padded.size(0), 15)

        for rate in [12]:

            padded = drnn._padinputs(x, rate)

            self.assertEqual(padded.size(0), 24)

        for rate in [18]:

            padded = drnn._padinputs(x, rate)

            self.assertEqual(padded.size(0), 18)

        self.assertEqual(padded[-1, 0, 0], 0)


class TestStackInputs(unittest.TestCase):
    def test(self):

        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.randn(16, 2, 13)

        if use_cuda:
            x = x.cuda()
            drnn.cuda()

        chunked = drnn._stack(x, 4)

        self.assertEqual(chunked.size(0), 4)
        self.assertEqual(chunked.size(1), 8)
        self.assertEqual(chunked.size(2), 13)

        self.assertTrue(torch.equal(x[0::4, 0, :], chunked[:, 0, :]))
        self.assertTrue(torch.equal(x[1::4, 0, :], chunked[:, 2, :]))
        self.assertTrue(torch.equal(x[2::4, 0, :], chunked[:, 4, :]))


class TestUnstackInputs(unittest.TestCase):
    def test(self):

        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.randn(16, 2, 13)

        if use_cuda:
            x = x.cuda()
            drnn.cuda()

        roundtrip = drnn._unstack(drnn._stack(x, 4), 4)

        self.assertTrue(torch.equal(drnn._unstack(drnn._stack(x, 4), 4), x))


class TestForward(unittest.TestCase):
    def test(self):

        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.randn(15, 2, 13)

        if use_cuda:
            x = x.cuda()
            drnn.cuda()

        outputs = drnn(torch.autograd.Variable(x))

        self.assertEqual(outputs.size(0), 15)
        self.assertEqual(outputs.size(1), 2)
        self.assertEqual(outputs.size(2), 64)


class TestReuse(unittest.TestCase):
    def test(self):
        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.autograd.Variable(torch.randn(15, 2, 13))

        if use_cuda:
            x = x.cuda()
            drnn.cuda()

        y = x.clone()

        drnn(x)

        self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()