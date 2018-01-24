import torch
import torch.nn as nn


use_cuda = torch.cuda.is_available()


class DilatedRNN(nn.Module):
    """
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer

    Inputs: input
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` for details.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** hidden state at last time point

    """
    def __init__(self, mode, input_size, dilations, hidden_sizes, dropout):

        super(DilatedRNN, self).__init__()

        assert len(hidden_sizes) == len(dilations)

        self.dilations = dilations
        self.cells = []
        next_input_size = input_size

        for hidden_size in hidden_sizes:
            self.cells.append(mode(input_size=next_input_size, hidden_size=hidden_size,
                              dropout=dropout, num_layers=1))
            next_input_size = hidden_size

    def cuda(self, device=None):
        for cell in self.cells:
            cell.cuda(device=device)

    def _padinputs(self, inputs, rate):

        num_steps = len(inputs)
        if num_steps % rate:
            dilated_num_steps = num_steps // rate + 1

            zeros_tensor = torch.zeros(dilated_num_steps * rate - num_steps,
                                       inputs.size(1),
                                       inputs.size(2))

            if use_cuda:
                zeros_tensor = zeros_tensor.cuda()

            zeros_tensor = torch.autograd.Variable(zeros_tensor)

            inputs = torch.cat((inputs, zeros_tensor))

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

        # remove padded zeros so output is [num_steps, batch_size, hidden_size]
        return outputs[:inputs.size(0)]

    def forward(self, x):
        for cell, dilation in zip(self.cells, self.dilations):
            x = self._dilated_RNN(cell, x, dilation)

        return x
