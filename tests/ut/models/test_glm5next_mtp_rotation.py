# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

import vllm_ascend.models.glm5next.mtp as mtp_module
from vllm_ascend.models.glm5next.mtp import (
    Glm5NextMTP,
    Glm5NextMultiTokenPredictorLayer,
)


def make_mtp(*, rot_enabled: bool = True):
    model = Glm5NextMTP.__new__(Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=0,
        mla_nope=False,
        qk_rope_head_dim=0,
    )
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 45
    model.model.num_mtp_layers = 1
    layer = nn.Module()
    layer.is_rot_used = rot_enabled
    layer.enorm = nn.LayerNorm(4, bias=False)
    if rot_enabled:
        layer.rot = nn.Linear(4, 4, bias=False)
    model.model.layers = nn.ModuleDict({"45": layer})
    return model


def test_root_rotation_weight_loads_into_first_mtp_layer():
    model = make_mtp()
    rotation = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    weights = [
        ("rot.weight", rotation),
        ("model.language_model.layers.45.enorm.weight", torch.full((4,), 2.0)),
    ]
    with patch.object(mtp_module, "fused_moe_make_expert_params_mapping", return_value=[]):
        loaded = model.load_weights(weights)

    assert "model.layers.45.rot.weight" in loaded
    torch.testing.assert_close(model.model.layers["45"].rot.weight, rotation)


def test_missing_required_rotation_weight_fails_cleanly():
    model = make_mtp()
    weights = [
        ("model.language_model.layers.45.enorm.weight", torch.full((4,), 2.0)),
    ]
    with (
        patch.object(mtp_module, "fused_moe_make_expert_params_mapping", return_value=[]),
        pytest.raises(ValueError, match="requires rot.weight"),
    ):
        model.load_weights(weights)


def test_rotation_is_applied_before_fused_hnorm():
    layer = Glm5NextMultiTokenPredictorLayer.__new__(Glm5NextMultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.is_rot_used = True
    layer.rot = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.rot.weight.copy_(2 * torch.eye(4))
    layer.enorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    layer.hnorm = SimpleNamespace(weight=torch.ones(4))
    layer.eh_proj = nn.Identity()

    class Block(nn.Module):
        def forward(self, *, positions, hidden_states, residual):
            return hidden_states, residual, None, None

    class Norm(nn.Module):
        def forward(self, hidden_states, residual=None):
            return hidden_states, None

    layer.mtp_block = Block()
    layer.shared_head = SimpleNamespace(norm=Norm())
    previous = torch.tensor([[0.5, -1.0, 2.0, 3.0]])
    captured = {}

    def fake_fused_eh_norm(positions, inputs_embeds, previous_hidden_states, *args):
        captured["previous"] = previous_hidden_states.clone()
        return torch.cat((inputs_embeds, previous_hidden_states), dim=-1)

    with patch.object(mtp_module, "fused_eh_norm", side_effect=fake_fused_eh_norm):
        layer(
            torch.tensor([2]),
            torch.tensor([10]),
            previous,
            inputs_embeds=torch.ones(1, 4),
        )

    torch.testing.assert_close(captured["previous"], 2 * previous)
