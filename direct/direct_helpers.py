import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from scipy.optimize import linprog


# Magic numbers from the internet
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


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


def get_avg_MNIST(sample_size:int=1000, loader:DataLoader=None):
    if loader is None:
        loader = DataLoader(MNIST('images/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
                                   ])),
                              batch_size=1, shuffle=True, pin_memory=False)
    avg = None
    for sample_num, (X, y) in enumerate(loader):
        if avg is None:
            avg = X
        else:
            avg = avg + X
        if sample_num == sample_size:
            break
    return avg / sample_num


def linprog_parameter(c_values, in_size):
    # supply out_size different values of c, each randomly sampled
    # for later matrices, mult in_size diff values through
    results = []
    for c in c_values:
        
        A_eq = torch.ones_like(c).expand((1, in_size)).detach().numpy()
        b_eq = torch.zeros((1,)).detach().numpy()
        c_np = c.detach().numpy()
        result = linprog(c_np, A_eq=A_eq, b_eq=b_eq, bounds=(-(1 / math.sqrt(in_size)), (1 / math.sqrt(in_size))))

        results.append(result.x)

    result_matrix = torch.stack(
        [torch.zeros(in_size).uniform_() * torch.tensor(x) for x in results]
    )
    
    return torch.nn.Parameter(result_matrix.to(dtype=torch.float32))