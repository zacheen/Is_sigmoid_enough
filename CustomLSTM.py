import torch
import torch.nn as nn

# ==========================================
# Custom LSTM Cell with ScaledSigmoid Gates
# ==========================================
# PyTorch's nn.LSTM hardcodes sigmoid in C++ for gate activations.
# To test ScaledSigmoid in LSTM gates, we must implement the cell manually.
#
# Standard LSTM cell equations:
#   i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)   <- input gate
#   f = sigmoid(W_if x + b_if + W_hf h + b_hf)   <- forget gate
#   g = tanh(W_ig x + b_ig + W_hg h + b_hg)      <- cell candidate
#   o = sigmoid(W_io x + b_io + W_ho h + b_ho)    <- output gate
#   c' = f * c + i * g
#   h' = o * tanh(c')
#
# ScaledSigmoid replaces sigmoid with: scale * sigmoid(x) + shift
# When scale=1.0, shift=0.0, this is identical to standard LSTM.
# ==========================================

class ScaledSigmoidLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, scale=1.0, shift=0.0):
        super(ScaledSigmoidLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale = scale
        self.shift = shift

        # Combined weight matrices for all 4 gates (i, f, g, o)
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

    def _scaled_sigmoid(self, x):
        return self.scale * torch.sigmoid(x) + self.shift

    def forward(self, x, state=None):
        """
        Args:
            x: (batch, input_size)
            state: tuple of (h, c), each (batch, hidden_size). None to init as zeros.
        Returns:
            (h_next, c_next): each (batch, hidden_size)
        """
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        else:
            h, c = state

        # Compute all 4 gates at once
        gates = self.ih(x) + self.hh(h)

        # Split into 4 chunks: input, forget, cell candidate, output
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        i = self._scaled_sigmoid(i_gate)   # input gate
        f = self._scaled_sigmoid(f_gate)   # forget gate
        g = torch.tanh(g_gate)             # cell candidate (tanh, not sigmoid)
        o = self._scaled_sigmoid(o_gate)   # output gate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ==========================================
# Multi-Layer Custom LSTM
# ==========================================
# Wraps ScaledSigmoidLSTMCell into a multi-layer LSTM
# with the same interface as nn.LSTM(batch_first=True).
# ==========================================

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, scale=1.0, shift=0.0):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(
                ScaledSigmoidLSTMCell(cell_input_size, hidden_size, scale=scale, shift=shift)
            )

    def forward(self, x, state=None):
        """
        Args:
            x: (batch, seq_len, input_size) — batch-first format
            state: tuple of (h_0, c_0), each (num_layers, batch, hidden_size). None to init as zeros.
        Returns:
            output: (batch, seq_len, hidden_size) — hidden state from last layer at each time step
            (h_n, c_n): each (num_layers, batch, hidden_size) — final states
        """
        batch_size, seq_len, _ = x.size()

        # Initialize states per layer
        if state is None:
            h_states = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                        for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                        for _ in range(self.num_layers)]
        else:
            h_0, c_0 = state
            h_states = [h_0[i] for i in range(self.num_layers)]
            c_states = [c_0[i] for i in range(self.num_layers)]

        # Collect outputs from the last layer
        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]  # (batch, input_size)
            for layer, cell in enumerate(self.cells):
                h_states[layer], c_states[layer] = cell(inp, (h_states[layer], c_states[layer]))
                inp = h_states[layer]  # output of this layer is input to next
            outputs.append(h_states[-1])

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)
        h_n = torch.stack(h_states, dim=0)    # (num_layers, batch, hidden_size)
        c_n = torch.stack(c_states, dim=0)    # (num_layers, batch, hidden_size)

        return output, (h_n, c_n)
