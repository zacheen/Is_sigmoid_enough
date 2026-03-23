import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from ScaledSigmoid import ScaledSigmoid
from util import replace_sigmoid_with_modified

# ==========================================
# Define Simple Neural Network Architecture
# ==========================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        seed = 42
        torch.manual_seed(seed)
        super(SimpleNN, self).__init__()
        # Simple two-layer fully connected network (e.g., 2x2 or 3x3)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = activation_fn
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = activation_fn
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x

# ==========================================
# 4. Training and Tracking Experiment Function
# ==========================================
def train_and_track(model, X, Y, optimizer, epochs=20000, lr=None):
    """
    Train and return the loss and mean absolute value of weights per epoch, 
    to observe whether weights grow excessively.
    """
    criterion = nn.MSELoss()
    if lr == None:
        if optimizer == optim.SGD:
            lr = 0.1
        else:
            lr = 0.01
    optimizer = optimizer(model.parameters(), lr=lr)
    
    loss_history = []
    weight_mag_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        # Record Loss
        loss_history.append(loss.item())
        
        # Record weight magnitude (using FC2's mean absolute weight as an example)
        fc2_weight_mag = model.fc2.weight.abs().mean().item()
        weight_mag_history.append(fc2_weight_mag)
        
    return loss_history, weight_mag_history

def compare(X, Y, node_size, optimizer, compare_type = 0):
    # compare_type = 0: original vs scaled sigmoid
    # compare_type = 1: original vs scaled sigmoid vs scaled sigmoid

    model_ori = SimpleNN(input_dim=node_size[0], hidden_dim=node_size[1], output_dim=node_size[2], activation_fn=nn.Sigmoid())
    loss_history_ori, weight_mag_history_ori = train_and_track(model_ori, X, Y, optimizer)
    
    model_1 = SimpleNN(input_dim=node_size[0], hidden_dim=node_size[1], output_dim=node_size[2], activation_fn=ScaledSigmoid(scale=1.01, shift=-0.005))
    loss_history_1, weight_mag_history_1 = train_and_track(model_1, X, Y, optimizer)

    model_2 = SimpleNN(input_dim=node_size[0], hidden_dim=node_size[1], output_dim=node_size[2], activation_fn=ScaledSigmoid(scale=1.05, shift=-0.025))
    loss_history_2, weight_mag_history_2 = train_and_track(model_2, X, Y, optimizer)

    model_3 = SimpleNN(input_dim=node_size[0], hidden_dim=node_size[1], output_dim=node_size[2], activation_fn=ScaledSigmoid(scale=1.1, shift=-0.05))
    loss_history_3, weight_mag_history_3 = train_and_track(model_3, X, Y, optimizer)

    # print final loss
    print("Original Loss     :", loss_history_ori[-1])
    print("ScaledSigmoid Loss:", loss_history_1[-1])
    print("ScaledSigmoid Loss:", loss_history_2[-1])
    print("ScaledSigmoid Loss:", loss_history_3[-1])

    # print the final weights and bias of the model
    print("Original Weights:",model_ori.fc1.weight.data, model_ori.fc2.weight.data)
    print("Original Bias:", model_ori.fc1.bias.data, model_ori.fc2.bias.data)
    print("ScaledSigmoid Weights:", model_1.fc1.weight.data, model_1.fc2.weight.data)
    print("ScaledSigmoid Bias:", model_1.fc1.bias.data, model_1.fc2.bias.data)
    print("ScaledSigmoid Weights:", model_2.fc1.weight.data, model_2.fc2.weight.data)
    print("ScaledSigmoid Bias:", model_2.fc1.bias.data, model_2.fc2.bias.data)
    print("ScaledSigmoid Weights:", model_3.fc1.weight.data, model_3.fc2.weight.data)
    print("ScaledSigmoid Bias:", model_3.fc1.bias.data, model_3.fc2.bias.data)

    # --- Plotting Result Graphs ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # First graph: Loss reduction speed
    ax1.plot(loss_history_ori, label="Original")
    ax1.plot(loss_history_1, label="ScaledSigmoid(scale=1.01, shift=-0.005)")
    ax1.plot(loss_history_2, label="ScaledSigmoid(scale=1.05, shift=-0.025)")
    ax1.plot(loss_history_3, label="ScaledSigmoid(scale=1.1, shift=-0.05)")
    ax1.set_title("Training Loss Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True)

    
    # Second graph: Weight magnitude changes
    ax2.plot(weight_mag_history_ori, label="Original")
    ax2.plot(weight_mag_history_1, label="ScaledSigmoid(scale=1.1, shift=-0.05)")
    if compare_type == 1:
        ax2.plot(weight_mag_history_2, label="ScaledSigmoid(scale=2.0, shift=-0.5)")
    ax2.set_title("Weight Magnitude Over Time (FC2 Layer)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Mean Absolute Weight")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    

    
# ==========================================
# Main Experiment Execution Section
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # < test case 0 >
    # if X is odd output 1, if X is even output 0 : failed(not able to achieve loss 0)
    # X = torch.randint(0, 10000, (10000, 1)).float()
    # Y = (X % 2).float()

    # < test case 1 >
    # if X > certain value output 1, else output 0 : success(able to achieve loss 0)
    # optimal solution 
        # original sigmoid
            # fc1's weight = 1, bias = -0.5
            # fc2's weight = big_num, bias = -(1/2)*big_num 
                # big_num should be large enough to make the loss close to 0
        # scaled sigmoid
            # fc1's weight = 1, bias = -0.5
            # fc2's weight = big_num, bias = -(1/2)*big_num 
                # 
    # experiment result : 
        # why the weight didn't explode?
            # since the loss is already close to 0, the gradient vanished
        # why Original Loss is lower ?
            # Now the loss include the error lower than 1 "and the error exceed 1" 
            # take scale=2.0, shift=-0.5 as an example, the output range is [-0.5, 1.5]
            # thus when the x bigger then certain value x_to_big, the output would exceed 1
                # thus from 0.5 ~ x_to_big it wants the weight to be bigger
                # but  from x_to_big ~ 1   it wants the weight to be smaller
            # Thus the gradient direction would be contradictory
            
    threshold = 0.5
    X = torch.rand(10000, 1).float()
    Y = (X > threshold).float()
    # for i in range(10): # checked
    #     print(X[i], Y[i])
    node_size = (1, 1, 1)
    optimizer = optim.Adam

    # # < test case 2 >
    # # Generate a set of normally distributed data, roughly in the range of -6 to 6 
    #     # to fix the gradient vanishing problem caused by raw values as large as 10000)
    # X = torch.randn(2000, 1) * 2
    # print(len(X))
    # # If X > 0 output 1.0, else output 0.0. This is a task that requires the model to output an absolute hard boundary (Hard 0/1)
    # Y = (X > 0).float()
    
    # # < test case 3 >
    # # Generate a definitive testcase: A strict Pulse wave (Hard Boundaries)
    # X = torch.linspace(-5, 5, 2000).view(-1, 1)
    # # Task: Output 1.0 only if X is between -1 and 1, else 0.0.
    # Y = ((X > -1) & (X < 1)).float()

    # << settings >>

    # 
    # < optimizer >
    # setting optimizer to SGD is easier to compare with the change in loss
    # optimizer = optim.SGD
    # using adam optimizer will make the weight explode
    # optimizer = optim.Adam

    compare(X, Y, node_size, optimizer, compare_type=1)

    # # ==========================================
    # # Appendix Experiment: Test replacing Sigmoid in an existing model
    # # ==========================================
    # print("Testing replacement of Sigmoid in an existing model...")
    # # Assume this is a model loaded from another framework or existing package
    # existing_model = nn.Sequential(
    #     nn.Conv2d(3, 16, 3),
    #     nn.Sigmoid(),
    #     nn.AdaptiveAvgPool2d(1),
    #     nn.Sigmoid()
    # )
    # print("=> Before replacement:", existing_model)
    
    # modified_model = replace_sigmoid_with_modified(copy.deepcopy(existing_model), scale=1.1, shift=-0.05)
    # print("\n=> After replacement:", modified_model)