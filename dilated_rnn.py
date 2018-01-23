import torch
import torch.nn as nn

class DilatedRNN(nn.Module):
    r"""Multilayer Dilated RNN.
    Args:
    mode: rnn type, 'RNN', 'LSTM', 'GRU'
    input_size: input size of the first layer
    dilations: list of dilations for each layer
    hidden_sizes: list of hidden sizes for rnn in each layer
    dropout: dropout prob.
    """
    def __init__(self, mode, input_size, dilations, hidden_sizes, dropout):
        super(DilatedRNN, self).__init__()

        assert len(hidden_sizes) == len(dilations)

        self.dilations = dilations
        self.cells = []
        next_input_size = input_size

        # TODO: have a single hidden size

        for hidden_size in hidden_sizes:
            if mode == "RNN":
                cell = nn.RNN(input_size=next_input_size, hidden_size=hidden_size,
                              dropout=dropout, num_layers=1)
            elif mode == "LSTM":
                cell = nn.LSTM(input_size=next_input_size, hidden_size=hidden_size,
                               dropout=dropout, num_layers=1)
            elif mode == "GRU":
                cell = nn.GRU(input_size=next_input_size, hidden_size=hidden_size,
                              dropout=dropout, num_layers=1)
            self.cells.append(cell)
            next_input_size = hidden_size


    """
    Args:
    inputs: [num_steps, batch_size, input_size]
    rate: integer
    output: [num_steps, batch_size, hidden_size]
    """
    def _padinputs(self, inputs, rate):

        num_steps = len(inputs)

        if num_steps % rate:
            zero_tensor = torch.zeros_like(inputs[0])

            dilated_num_steps = num_steps // rate + 1
            for _ in range(dilated_num_steps * rate - num_steps):
                inputs = torch.cat((inputs, zero_tensor.unsqueeze(0)))

        return inputs

    def _stack(self, x, rate):
        tostack = [x[i::rate] for i in range(rate)]
        stacked = torch.cat(tostack, 1)

        return stacked

    def _unstack(self, x, rate):
        outputs = x.view(x.size(0) * rate, -1, x.size(2))

        return outputs

    def _dilated_RNN(self, cell, inputs, rate):

        # add zeros to last few time steps if not zero mod rate
        padded_inputs = self._padinputs(inputs, rate)

        # dilated_inputs is of size [dilated_num_steps, rate*batch_size, input_size]
        dilated_inputs = self._stack(padded_inputs, rate)

        # output is of size [dilated_num_steps, rate*batch_size, hidden_size]
        dilated_outputs, _ = cell(dilated_inputs)

        # reshape it to [dilated_num_steps*rate, batch_size, hidden_size]
        outputs = self._unstack(dilated_outputs, rate)
        #outputs = dilated_outputs.view(dilated_num_steps*rate, -1, dilated_outputs.shape[2])

        # remove padded zeros so output is [num_steps, batch_size, hidden_size]
        return outputs[:inputs.size(0)]

    """
    Args:
    input: [num_steps, batch_size, input_size]
    output: [num_steps, batch_size, hidden_size]
    """
    def forward(self, x):
        for cell, dilation in zip(self.cells, self.dilations):
            x = self._dilated_RNN(cell, x, dilation)

        return x
