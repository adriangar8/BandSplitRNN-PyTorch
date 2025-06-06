import torch
import torch.nn as nn
from model.modules.se_layer import SELayer1D


class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)

        out = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]

        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        out = self.rnn(out)[0]  # [BK, T, H]    [BT, K, H]
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        return x


class BandSequenceModelModule(nn.Module):
    
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(self, input_dim_size: int, hidden_dim_size: int, rnn_type: str = 'lstm',
                bidirectional: bool = True, num_layers: int = 12):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])

        for _ in range(num_layers):
            rnn_across_t = RNNModule(
                input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            rnn_across_k = RNNModule(
                input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            self.bsrnn.append(
                nn.Sequential(rnn_across_t, rnn_across_k)
            )
        
        # -----------------------------------------------------------------------------
        
        # -- SE layers here: one per RNNModule (2 per layer) --
        self.se_layers = nn.ModuleList([
            SELayer1D(input_dim_size) for _ in range(num_layers * 2)
        ])
        
        # -----------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        se_idx = 0
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)  # shape: [B, K, T, N]

            # Rearrange x to [B*K, N, T] for SE (channels = features dimension)
            B, K, T, N = x.shape
            x_ = x.view(B * K, T, N).permute(0, 2, 1).contiguous()

            # Apply SE layer and permute back
            x_ = self.se_layers[se_idx](x_)
            se_idx += 1
            x_ = x_.permute(0, 2, 1).contiguous().view(B, K, T, N)

            # Apply SE after first RNNModule (across time)

            # The second RNNModule (across subbands) also gets its own SE layer,
            # so repeat the same on x

            # Apply second SE layer
            x_ = x.view(B, K, T, N).permute(0, 2, 1, 3).contiguous()
            # Now shape: [B, T, K, N]
            x_ = x_.view(B * T, K, N).permute(0, 2, 1).contiguous()
            x_ = self.se_layers[se_idx](x_)
            se_idx += 1
            x_ = x_.permute(0, 2, 1).contiguous().view(B, T, K, N)
            x = x_.permute(0, 2, 1, 3).contiguous()

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, k_subbands, t_timesteps, input_dim = 4, 41, 259, 128
    in_features = torch.rand(batch_size, k_subbands, t_timesteps, input_dim).to(device)

    cfg = {
        "t_timesteps": t_timesteps,
        "input_dim_size": 128,
        "hidden_dim_size": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 12
    }
    model = BandSequenceModelModule(**cfg).to(device)
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(f"In: {in_features.shape}\nOut: {out_features.shape}")
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")
