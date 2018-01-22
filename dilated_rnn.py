import torch
import torch.nn as nn

class DilatedRNN(nn.Module):
    r"""Multilayer Dilated RNN.
    Args:
    
    """
    def __init__(self, mode, input_size, 
                 dilations, hidden_sizes):
        super(DilatedRNN, self).__init__()

        assert(len(hidden_sizes)==len(dilations))
        self.cells = []
        for hidden_size in hidden_sizes:
            if mode == "RNN":
                cell = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1)
            elif mode == "LSTM":
                cell = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
            elif mode == "GRU":
                cell = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)
            self.cells.append(cell)
        

    """
    Args:
    inputs: [num_steps, batch_size, input_size]
    """
    def _dilated_RNN(self, cell, inputs, rate):
        num_steps = len(inputs)

        if (num_steps % rate):
            # Zero padding with tensor of size [batch_size, input_size]
            zero_tensor = torch.zeros_like(inputs[0])

            dilated_num_steps = num_steps // rate + 1
            for _ in range(dilated_num_steps*rate - num_steps):
                torch.cat((inputs, zero_tensor.unsqueeze(0)), out=inputs)
        else:
            dilated_num_steps = num_steps // rate

        # E.g. if num_steps is 5, rate is 2 and inputs = [x1, x2, x3, x4, x5]
        # we do zero padding --> [x1, x2, x3, x4, x5, 0]
        # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
        # where the length is dilated_num_steps
        dilated_inputs = torch.cat((inputs[i * rate:(i + 1) * rate] for i in range(dialated_n_steps)), dim=0)

        # dilated_inputs needs to be of size [dilated_num_steps, batch_size, input_size]
        dilated_outputs, _ = cell(dilated_inputs)
        
        # reshape dilated_outputs back to [rate, batch_size, input_size]
        

        # remove padded zeros so output is [??, batch_size, input_size]

        # return output

    """
    Args: 
    input: [num_steps, batch_size, input_size] 
    """
    def forward(self, inputs):
        x = inputs.clone()

        for cell, dilation in zip(self.cells, self.dilations)
            x = _dilated_RNN(cell, x, dilation)

        return x

