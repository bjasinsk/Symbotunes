from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.base import BaseModel, OutputType
from .generator import BarGenerator
from .discriminator import Discriminator
from .generator import TemporalGenerator


class MuseGAN(BaseModel, pl.LightningModule):
    def __init__(
        self,
        config: dict[str, Any],
        temp_generator: list[TemporalGenerator] | None = None,
        discriminator: Discriminator | None = None,
    ) -> None:
        super(MuseGAN, self).__init__()
        self.save_hyperparameters(ignore=['generator', 'temp_generator', 'discriminator'])
        self.discriminator = discriminator if discriminator else Discriminator()
        self.g_temp = temp_generator if temp_generator else TemporalGenerator()
        self.config = config
        self.g_temp_tracks = nn.ModuleList([TemporalGenerator() for _ in range(config.get("g_temp_number", 5))])
        self.bar_generators = nn.ModuleList([BarGenerator() for _ in range(config.get("bar_generator_number", 5))])
        training_config = config.get("training", {})
        self.max_steps = training_config.get("max_steps", 1000)
        self.log_steps = training_config.get("log_steps", 50)
        self.save_checkpoints_steps = training_config.get("save_steps", 500)
        self.d_steps = training_config.get("d_steps", 1)
        self.g_steps = training_config.get("g_steps", 1)
        self.warmup_steps = training_config.get("warmup_steps", 20)
        self.lambda_gp = training_config.get("lambda_gp", 10.0)
        self.lr = config.get("learning_rate", 1e-4)
        self.beta1 = config.get("beta1", 0.5)
        self.beta2 = config.get("beta2", 0.999)
        self.automatic_optimization = False

    def forward(self, batch_size = 1):
        bars = []
        # First stage
        z_t = self.g_temp(torch.rand(batch_size, self.g_temp.latent_dim).to(self.device))
        z_t_tracks = [g_temp_track(torch.rand(batch_size, g_temp_track.latent_dim).to(self.device)) for g_temp_track in self.g_temp_tracks]
        assert all(z_t_track.size(2) == z_t.size(2) for z_t_track in z_t_tracks)

        # Second stage
        generated_bars = []
        for i in range(z_t.size(2)):
            z_t_squeezed = torch.squeeze(z_t[:, :, i]) if z_t.size(0) > 1 else z_t[:, :, i]
            z = torch.rand(z_t_squeezed.shape).to(self.device)
            z_concat = torch.cat([z, z_t_squeezed], dim=1)
            
            z_t_tracks_squeezed = [torch.squeeze(z_t_track[:, :, i]) if z_t_track.size(0) > 1 else z_t_track[:, :, i] for z_t_track in z_t_tracks]
            z_tracks = [torch.rand(z_t_track_squeezed.shape).to(self.device) for z_t_track_squeezed in z_t_tracks_squeezed]
            z_tracks_concat = [torch.cat([z_track, z_t_track_squeezed], dim=1) for (z_track, z_t_track_squeezed) in zip(z_tracks, z_t_tracks_squeezed)]
            z_tracks_concat_concat = [torch.cat([z_concat, z_track_concat], dim=1) for z_track_concat in z_tracks_concat]

            bars = [bar_generator_i(z_tracks_concat_concat_i.unsqueeze(-1).unsqueeze(-1)) for z_tracks_concat_concat_i, bar_generator_i in zip(z_tracks_concat_concat, self.bar_generators)]
            generated_bars.append(bars)
        
        return generated_bars
    
    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor] | torch.Tensor:
        bars_prepared = [torch.cat(bar, dim=1).unsqueeze(2) for bar in self.forward(batch_size)]
        return torch.cat(bars_prepared, dim=2)
    
    @staticmethod
    def get_produced_type() -> OutputType:
        return OutputType.PYPiano

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        optimizers = self.optimizers()
        d_optimizer = optimizers[0]
        g_temp_optimizer = optimizers[1]
        g_temp_optimizers = optimizers[2 : 2 + len(self.g_temp_tracks)]
        bar_generators_optimizers = optimizers[2 + len(self.g_temp_tracks) :]

        real = batch[0]
        BATCH, BAR, PITCH, NOTE, TRACK = real.shape

        real = real.permute(0, 4, 1, 2, 3)
        real = 2.0 * real - 1.0 

        bars = self.forward(BATCH)
        bars_prepared = [torch.cat(bar, dim=1).unsqueeze(2) for bar in bars]
        fake = torch.cat(bars_prepared, dim=2)
        dis_real = self.discriminator(real)  # B x 5(track) x 4(bar) x 96(time step) x 84(note)
        dis_fake = self.discriminator(fake.detach())

        grad_penalty = self._get_gradient_penalty(real, fake.detach())
        dis_loss = self._dis_loss(dis_fake, dis_real, grad_penalty)

        d_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(dis_loss)
        d_optimizer.step()
        
        g_temp_optimizer.zero_grad(set_to_none=True)

        for opt in g_temp_optimizers:
            opt.zero_grad(set_to_none=True)

        for opt in bar_generators_optimizers:
            opt.zero_grad(set_to_none=True)

        bars = self(BATCH)
        bars_prepared = [torch.cat(bar, dim=1).unsqueeze(2) for bar in bars]
        fake = torch.cat(bars_prepared, dim=2)
        gen_loss = self._gen_loss(fake)
        
        self.manual_backward(gen_loss)
        g_temp_optimizer.step()
        
        for g_temp_opt in g_temp_optimizers:
            g_temp_opt.step()

        for bar_generator_opt in bar_generators_optimizers:
            bar_generator_opt.step()

        self.log_dict({"Loss/D": dis_loss, "Loss/G": gen_loss, "Loss/GP": grad_penalty}, on_epoch=True, prog_bar=True)

    def _get_gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        eps = torch.rand(real.size(0), 1, 1, 1, 1, device=self.device)
        interpolated_x = eps * real + (1.0 - eps) * fake
        interpolated_x.requires_grad_(True)

        dis_inter = self.discriminator(interpolated_x)
        grads = torch.autograd.grad(
            outputs=dis_inter, inputs=interpolated_x, grad_outputs=torch.ones_like(dis_inter), create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grads = grads.reshape(real.size(0), -1).norm(2, dim=1)
        gp = ((grads - 1.0) ** 2).mean() * self.lambda_gp

        return gp

    def _dis_loss(self, fake: torch.Tensor, real: torch.Tensor, gp: torch.tensor) -> torch.Tensor:
        return fake.mean() - real.mean() + gp

    def _gen_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return -self.discriminator(fake).mean()

    def configure_optimizers(self) -> list:
        optimizers = [
            torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)),
            torch.optim.Adam(self.g_temp.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        ]
        optimizers.extend([torch.optim.Adam(g_temp_track.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)) for g_temp_track in self.g_temp_tracks])
        optimizers.extend([torch.optim.Adam(bar_generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)) for bar_generator in self.bar_generators])
        return optimizers
