import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ReluCount(nn.ReLU):
    """
        Intended to behave identically to nn.ReLU, except for the zero counting.
        To disable zero counting, set count_zeros to False.
    """
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.zero_count = 0
        self.count_zeros = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        true_result = F.relu(input, inplace=self.inplace)
        if self.count_zeros == True:
            for entry in torch.flatten(true_result):
                if entry.item() == 0:
                    self.zero_count += 1
        return true_result
    
    def reset_count(self):
        self.zero_count = 0
        self.count_zeros = False
    
class MnistReluCountModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_count = ReluCount()
        # mnist images are 1x28x28, so flattened they will have a length of 28*28=784
        self.fc1 = nn.Linear(784, 750)
        self.fc2 = nn.Linear(750, 320)
        self.fc3 = nn.Linear(320, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten batches 2D images to 1D vectors
        x = self.relu_count(self.fc1(x))
        x = self.relu_count(self.fc2(x))
        x = self.relu_count(self.fc3(x))
        x = self.relu_count(self.fc4(x))
        return x
    
    def begin_count(self):
        """
        Clears internal count of ReLU zeros and prepares to count again
        """
        self.relu_count.reset_count()
        self.relu_count.count_zeros = True

    def get_activation_count(self):
        """
        Returns number of times ReLU has activated since begin_count was called
        """
        return self.relu_count.zero_count
    
    def reset_count(self):
        """
        Set internal ReLU activation count to 0 and disable counting
        """
        self.relu_count.reset_count()
    
def count_relu_0s(dataloader: DataLoader, model: MnistReluCountModel):
    model.eval()
    model.begin_count()
    for (X, y) in dataloader:
        model(X)
    num_zeros = model.get_activation_count()
    model.reset_count()
    return num_zeros