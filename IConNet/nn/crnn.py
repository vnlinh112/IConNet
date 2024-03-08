import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, n_input=1, n_output=2):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 39, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=39),
            nn.Conv2d(39, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Flatten()
        )
        self.n_output_channel = 256
        self.lstm = nn.LSTM(
            input_size=self.n_output_channel, 
            hidden_size=64, 
            num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, self.n_output)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 