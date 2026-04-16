import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CliffordInteraction(nn.Module):
    def __init__(self, dim, cli_mode="full", ctx_mode="diff", shifts=None):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.act = nn.SiLU()
        self.shifts = [shift for shift in (shifts or [1, 2]) if shift < dim]
        self.branch_dim = dim * len(self.shifts)

        if self.cli_mode == "full":
            cat_dim = self.branch_dim * 2
        elif self.cli_mode in ["wedge", "inner"]:
            cat_dim = self.branch_dim
        else:
            raise ValueError(f"Invalid cli_mode: {self.cli_mode}")

        self.proj = nn.Conv2d(cat_dim, dim, kernel_size=1)

    def forward(self, z1, z2):
        if self.ctx_mode == "diff":
            context = z2 - z1
        elif self.ctx_mode == "abs":
            context = z2
        else:
            raise ValueError(f"Invalid ctx_mode: {self.ctx_mode}")

        features = []
        for shift in self.shifts:
            context_shifted = torch.roll(context, shifts=shift, dims=1)
            if self.cli_mode in ["wedge", "full"]:
                z1_shifted = torch.roll(z1, shifts=shift, dims=1)
                wedge = z1 * context_shifted - context * z1_shifted
                features.append(wedge)
            if self.cli_mode in ["inner", "full"]:
                inner = self.act(z1 * context_shifted)
                features.append(inner)

        return self.proj(torch.cat(features, dim=1))


class CliffordBlock(nn.Module):
    def __init__(
        self,
        dim,
        cli_mode="full",
        ctx_mode="diff",
        shifts=None,
        drop_path_rate=0.1,
        init_values=1e-5,
    ):
        super().__init__()
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.norm = LayerNorm2d(dim)
        self.interaction = CliffordInteraction(
            dim,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            shifts=shifts or [1, 2],
        )
        self.gate = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_norm = self.norm(x)
        z_state = self.get_state(x_norm)
        z_context = self.get_context(x_norm)
        mixed = self.interaction(z_state, z_context)

        gate = torch.sigmoid(self.gate(torch.cat([x_norm, mixed], dim=1)))
        x_mixed = F.silu(x_norm) + gate * mixed
        return shortcut + self.drop_path(self.gamma * x_mixed)


class GeometricStem(nn.Module):
    def __init__(self, in_chans=1, embed_dim=128, patch_size=1):
        super().__init__()
        if patch_size == 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=1, padding=1, bias=False),
            )
        elif patch_size == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class CliffordNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        patch_size=1,
        embed_dim=64,
        cli_mode="full",
        ctx_mode="diff",
        shifts=None,
        depth=6,
        drop_path_rate=0.1,
        in_chans=1,
    ):
        super().__init__()
        self.patch_embed = GeometricStem(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        block_drop_rates = [value.item() for value in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                CliffordBlock(
                    dim=embed_dim,
                    cli_mode=cli_mode,
                    ctx_mode=ctx_mode,
                    shifts=shifts or [1, 2],
                    drop_path_rate=block_drop_rates[idx],
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[-2, -1])
        return self.norm(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.dropout(self.act(self.bn1(self.conv1(x))))
        x = self.dropout(self.act(self.bn2(self.conv2(x))))
        return self.act(x + residual)


class LightweightTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, out_dim=32, dropout=0.1):
        super().__init__()
        self.block1 = TemporalBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(hidden_dim, out_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        h_temp = x.transpose(1, 2).contiguous()
        z_temp = self.pool(x).squeeze(-1)
        return h_temp, z_temp

    def forward(self, x):
        _, z_temp = self.forward_features(x)
        return z_temp


class DeepHPM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, 1, batch_first=True)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(attn_out + x)
        hidden = self.fc1(x)
        hidden = self.activation(hidden + x)
        return self.fc2(hidden)


class TCNPINNRegularizer(nn.Module):
    def __init__(self, hidden_dim, dt=None):
        super().__init__()
        self.dt = dt
        self.deep_hpm = DeepHPM(hidden_dim, hidden_dim)

    def resolve_dt(self, hidden_states):
        if self.dt is not None:
            return float(self.dt)
        return 1.0

    def forward(self, hidden_states):
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected hidden_states with shape [B, L, C], got {tuple(hidden_states.shape)}")

        if hidden_states.size(1) < 2:
            zero = hidden_states.new_zeros(())
            return zero, {"physics_residual_rms": zero.detach()}

        dt = self.resolve_dt(hidden_states)
        current_states = hidden_states[:, :-1, :]
        hidden_derivative = (hidden_states[:, 1:, :] - hidden_states[:, :-1, :]) / dt
        predicted_derivative = self.deep_hpm(current_states)
        residual = hidden_derivative - predicted_derivative
        physics_loss = residual.pow(2).mean()
        residual_rms = residual.pow(2).mean().sqrt()
        return physics_loss, {"physics_residual_rms": residual_rms.detach()}


class FusionModel(nn.Module):
    def __init__(
        self,
        seq_input_dim,
        gpgl_in_chans,
        num_classes=2,
        patch_size=1,
        embed_dim=32,
        depth=4,
        drop_path_rate=0.3,
        tcn_hidden_dim=32,
        tcn_out_dim=32,
        fusion_hidden_dim=32,
        aux_hidden_dim=32,
        classifier_dropout=0.1,
        phys_dt=None,
    ):
        super().__init__()
        self.clifford_encoder = CliffordNet(
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=gpgl_in_chans,
            depth=depth,
            drop_path_rate=drop_path_rate,
        )
        self.clifford_encoder.head = nn.Identity()

        self.temporal_encoder = LightweightTCN(
            input_dim=seq_input_dim,
            hidden_dim=tcn_hidden_dim,
            out_dim=tcn_out_dim,
            dropout=classifier_dropout,
        )
        self.temporal_classifier = nn.Sequential(
            nn.Linear(tcn_out_dim, aux_hidden_dim),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(aux_hidden_dim, num_classes),
        )
        self.physics_regularizer = TCNPINNRegularizer(hidden_dim=tcn_out_dim, dt=phys_dt)
        self.clifford_classifier = nn.Sequential(
            nn.Linear(embed_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        fusion_dim = embed_dim + tcn_out_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def encode_temporal(self, seq_x):
        return self.temporal_encoder.forward_features(seq_x.transpose(1, 2))

    def encode_clifford(self, gpgl_x):
        return self.clifford_encoder.forward_features(gpgl_x)

    def forward_temporal_branch(self, seq_x):
        h_temp, z_temp = self.encode_temporal(seq_x)
        logits = self.temporal_classifier(z_temp)
        return logits, {"h_temp": h_temp, "z_temp": z_temp}

    def forward_clifford_branch(self, gpgl_x):
        z_cliff = self.encode_clifford(gpgl_x)
        logits = self.clifford_classifier(z_cliff)
        return logits, {"z_cliff": z_cliff}

    def forward_fusion_head(self, z_temp, z_cliff, detach_temp=False):
        if detach_temp:
            z_temp = z_temp.detach()
        fused = torch.cat([z_temp, z_cliff], dim=1)
        return self.fusion_classifier(fused)

    def compute_physics_loss(self, h_temp):
        return self.physics_regularizer(h_temp)

    def forward(self, seq_x, gpgl_x=None):
        if gpgl_x is None:
            seq_x, gpgl_x = seq_x
        _, temporal_outputs = self.forward_temporal_branch(seq_x)
        _, clifford_outputs = self.forward_clifford_branch(gpgl_x)
        return self.forward_fusion_head(temporal_outputs["z_temp"], clifford_outputs["z_cliff"])

    def temporal_parameters(self):
        for module in (self.temporal_encoder, self.temporal_classifier, self.physics_regularizer):
            yield from module.parameters()

    def clifford_parameters(self):
        for module in (self.clifford_encoder, self.clifford_classifier):
            yield from module.parameters()

    def fusion_parameters(self):
        yield from self.fusion_classifier.parameters()

    def freeze_temporal_branch(self):
        for param in self.temporal_parameters():
            param.requires_grad = False


def build_model(cfg, seq_input_dim, gpgl_input_channels):
    embed_dim = int(cfg.embed_dim)
    tcn_hidden_dim = int(cfg.tcn_hidden_dim)
    tcn_out_dim = int(cfg.tcn_out_dim)
    fusion_hidden_dim = int(cfg.fusion_hidden_dim)
    aux_hidden_dim = int(cfg.aux_hidden_dim)
    classifier_dropout = float(cfg.fusion_dropout)

    return FusionModel(
        seq_input_dim=seq_input_dim,
        gpgl_in_chans=gpgl_input_channels,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
        embed_dim=embed_dim,
        depth=cfg.depth,
        drop_path_rate=cfg.drop_path_rate,
        tcn_hidden_dim=tcn_hidden_dim,
        tcn_out_dim=tcn_out_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        aux_hidden_dim=aux_hidden_dim,
        classifier_dropout=classifier_dropout,
        phys_dt=cfg.phys_dt,
    )
