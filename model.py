from torch import nn


class VPLSTM(nn.Module):
    def __init__(self, input_size, hid_size, layers, output_size):
        super(VPLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hid_size,
            layers,
        )
        self.fc = nn.Linear(hid_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
