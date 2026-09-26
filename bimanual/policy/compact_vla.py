"""Compact Redwood-inspired research policy.

This is an original experimental architecture based only on public design
ideas: shared multimodal tokens, chunked actions, conditional flow matching,
and auxiliary physical-prediction heads. It is not a reproduction of NEO or
of 1X's proprietary Redwood training stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bimanual.policy.types import PolicyOutput


@dataclass(frozen=True)
class CompactVLAConfig:
    state_dim: int = 12
    action_dim: int = 12
    chunk_size: int = 20
    # The research preset is approximately 100M parameters. The smoke preset
    # below is the default for local plumbing and tiny-overfit checks.
    hidden_dim: int = 768
    num_layers: int = 10
    num_heads: int = 12
    feedforward_dim: int = 5120
    vocabulary_size: int = 256
    max_language_tokens: int = 96
    max_history: int = 4
    num_cameras: int = 3
    num_phases: int = 8
    num_relations: int = 8
    num_events: int = 8
    context_dim: int = 32
    vision_grid_size: int = 2
    dropout: float = 0.1

    @classmethod
    def research_160m(cls, state_dim: int = 14, action_dim: int = 14) -> "CompactVLAConfig":
        return cls(state_dim=state_dim, action_dim=action_dim,
                   num_layers=15, feedforward_dim=5312)

    @classmethod
    def smoke(cls, state_dim: int = 12, action_dim: int = 12) -> "CompactVLAConfig":
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.0,
        )


class ByteTokenizer:
    """Deterministic dependency-free tokenizer for the first research pass."""

    def __init__(self, max_tokens: int = 96) -> None:
        self.max_tokens = max_tokens

    def __call__(self, texts: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        tokens = torch.zeros((len(texts), self.max_tokens), dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            raw = text.encode("utf-8")[: self.max_tokens]
            if raw:
                tokens[row, : len(raw)] = torch.tensor(list(raw), dtype=torch.long, device=device)
        return tokens


class SharedVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(grid_size),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images)
        return features.flatten(2).transpose(1, 2)


class CompactVLA(nn.Module):
    """Fuses image history, language, proprioception, memory and arm context."""

    def __init__(self, config: CompactVLAConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.vision = SharedVisionEncoder(h, config.vision_grid_size)
        self.text_embedding = nn.Embedding(config.vocabulary_size, h, padding_idx=0)
        self.state_projection = nn.Linear(config.state_dim, h)
        self.context_projection = nn.Linear(config.context_dim, h)
        self.camera_embedding = nn.Embedding(config.num_cameras, h)
        self.history_embedding = nn.Embedding(config.max_history, h)
        self.patch_embedding = nn.Embedding(config.vision_grid_size**2, h)
        self.modality_embedding = nn.Embedding(4, h)
        layer = nn.TransformerEncoderLayer(
            h, config.num_heads, config.feedforward_dim, config.dropout,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(h))

        self.action_query = nn.Parameter(torch.randn(config.chunk_size, h) * 0.02)
        self.flow_input = nn.Linear(config.action_dim, h)
        self.flow_time = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.flow_output = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.action_dim))

        self.phase_head = nn.Linear(h, config.num_phases)
        self.progress_head = nn.Sequential(nn.Linear(h, 1), nn.Sigmoid())
        self.grounding_head = nn.Sequential(nn.Linear(h, 8), nn.Sigmoid())
        self.relation_head = nn.Linear(h, config.num_relations)
        self.holding_head = nn.Linear(h, 2)
        self.event_head = nn.Linear(h, config.num_events)

    def _encode(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # images: B, history, camera, C, H, W
        b, history, cameras, channels, height, width = images.shape
        if history > self.config.max_history or cameras != self.config.num_cameras:
            raise ValueError(
                f"expected <= {self.config.max_history} history and {self.config.num_cameras} cameras; "
                f"got {history} and {cameras}"
            )
        flat = images.reshape(b * history * cameras, channels, height, width)
        patches = self.config.vision_grid_size**2
        visual = self.vision(flat).reshape(b, history, cameras, patches, -1)
        camera_ids = torch.arange(cameras, device=images.device)
        history_ids = torch.arange(history, device=images.device)
        patch_ids = torch.arange(patches, device=images.device)
        visual = visual + self.camera_embedding(camera_ids)[None, None, :, None]
        visual = visual + self.history_embedding(history_ids)[None, :, None, None]
        visual = visual + self.patch_embedding(patch_ids)[None, None, None]
        visual = visual.reshape(b, history * cameras * patches, -1) + self.modality_embedding.weight[0]

        language_mask = language_tokens.ne(0).unsqueeze(-1)
        language_sum = (self.text_embedding(language_tokens) * language_mask).sum(dim=1, keepdim=True)
        language = language_sum / language_mask.sum(dim=1, keepdim=True).clamp_min(1)
        language = language + self.modality_embedding.weight[1]
        state_token = self.state_projection(state).unsqueeze(1) + self.modality_embedding.weight[2]
        tokens = [visual, language, state_token]
        if context is not None:
            tokens.append(self.context_projection(context).unsqueeze(1) + self.modality_embedding.weight[3])
        encoded = self.transformer(torch.cat(tokens, dim=1))
        return encoded, encoded.mean(dim=1)

    def predict_velocity(
        self,
        latent: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = self.flow_input(noisy_actions)
        action_tokens = action_tokens + self.action_query[None]
        action_tokens = action_tokens + self.flow_time(time).unsqueeze(1)
        return self.flow_output(action_tokens + latent.unsqueeze(1))

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        context: torch.Tensor | None = None,
        noisy_actions: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> PolicyOutput:
        _tokens, latent = self._encode(images, state, language_tokens, context)
        if noisy_actions is None:
            noisy_actions = torch.zeros(
                state.shape[0], self.config.chunk_size, self.config.action_dim,
                device=state.device, dtype=state.dtype,
            )
        if flow_time is None:
            flow_time = torch.zeros(state.shape[0], 1, device=state.device, dtype=state.dtype)
        actions = self.predict_velocity(latent, noisy_actions, flow_time)
        return self._heads(latent, actions)

    def _heads(self, latent: torch.Tensor, actions: torch.Tensor) -> PolicyOutput:
        return PolicyOutput(
            actions=actions,
            phase_logits=self.phase_head(latent),
            progress=self.progress_head(latent),
            grounding=self.grounding_head(latent).reshape(-1, 4, 2),
            relation_logits=self.relation_head(latent),
            holding_logits=self.holding_head(latent),
            event_logits=self.event_head(latent),
        )

    def flow_matching_loss(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        target_actions: torch.Tensor,
        context: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _tokens, latent = self._encode(images, state, language_tokens, context)
        return self._flow_loss(latent, target_actions, action_is_pad)

    def flow_and_heads(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        target_actions: torch.Tensor,
        context: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PolicyOutput]:
        """Flow loss and auxiliary head outputs from one shared encoding."""
        _tokens, latent = self._encode(images, state, language_tokens, context)
        loss = self._flow_loss(latent, target_actions, action_is_pad)
        return loss, self._heads(latent, target_actions)

    def _flow_loss(
        self,
        latent: torch.Tensor,
        target_actions: torch.Tensor,
        action_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        noise = torch.randn_like(target_actions)
        time = torch.rand(target_actions.shape[0], 1, device=target_actions.device)
        interpolated = (1 - time[:, :, None]) * noise + time[:, :, None] * target_actions
        target_velocity = target_actions - noise
        predicted = self.predict_velocity(latent, interpolated, time)
        error = nn.functional.mse_loss(predicted, target_velocity, reduction="none")
        if action_is_pad is not None:
            valid = (~action_is_pad.bool()).unsqueeze(-1).to(error.dtype)
            return (error * valid).sum() / (valid.sum() * error.shape[-1]).clamp_min(1)
        return error.mean()

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        context: torch.Tensor | None = None,
        integration_steps: int = 8,
    ) -> PolicyOutput:
        _tokens, latent = self._encode(images, state, language_tokens, context)
        actions = torch.randn(
            state.shape[0], self.config.chunk_size, self.config.action_dim,
            device=state.device, dtype=state.dtype,
        )
        dt = 1.0 / integration_steps
        for step in range(integration_steps):
            time = torch.full((state.shape[0], 1), step / integration_steps, device=state.device)
            actions = actions + dt * self.predict_velocity(latent, actions, time)
        output = self.forward(images, state, language_tokens, context, actions, torch.ones_like(time))
        output.actions = actions
        return output

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
