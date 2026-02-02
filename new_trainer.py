import torch
from torch import nn
from torch.optim import Adam
from contextlib import nullcontext

class RNACDMTrainer(nn.Module):
    def __init__(
        self,
        imagen,
        lr=1e-4,
        device=None
    ):
        super().__init__()

        self.imagen = imagen
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imagen.to(self.device)

        # assume exactly one UNet
        self.unet = self.imagen.unets[0]
        self.optimizer = Adam(self.unet.parameters(), lr=lr)

        self.global_step = 0
        self.is_main = True   # single-process assumption

    # --------------------------------------------------
    # forward / training step
    # --------------------------------------------------
    def forward(
        self,
        images,
        *,
        methyl_embeds=None,
        unet_number=1,
        max_batch_size=None
    ):
        self.imagen.train()

        images = images.to(self.device)
        if methyl_embeds is not None:
            methyl_embeds = methyl_embeds.to(self.device)

        total_loss = 0.0

        # optional chunking
        if max_batch_size is None or images.shape[0] <= max_batch_size:
            loss = self.imagen(
                images,
                rna_embeds=methyl_embeds,
                unet_number=1
            )
            total_loss = loss
            loss.backward()
        else:
            batch_size = images.shape[0]
            for i in range(0, batch_size, max_batch_size):
                img_chunk = images[i:i+max_batch_size]
                rna_chunk = None if methyl_embeds is None else methyl_embeds[i:i+max_batch_size]

                loss = self.imagen(
                    img_chunk,
                    rna_embeds=rna_chunk,
                    unet_number=1
                )

                frac = img_chunk.shape[0] / batch_size
                (loss * frac).backward()
                total_loss += loss.detach() * frac

        return float(total_loss.item())

    __call__ = forward

    # --------------------------------------------------
    # optimizer step
    # --------------------------------------------------
    def update(self, unet_number=1):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    # --------------------------------------------------
    # sampling
    # --------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size,
        rna_embeds=None,
        cond_scale=1.0,
        stop_at_unet_number=1,
        return_pil_images=False
    ):
        self.imagen.eval()

        if rna_embeds is not None:
            rna_embeds = rna_embeds.to(self.device)

        return self.imagen.sample(
            batch_size=batch_size,
            rna_embeds=rna_embeds,
            cond_scale=cond_scale,
            stop_at_unet_number=stop_at_unet_number,
            return_pil_images=return_pil_images,
            device=self.device
        )

    # --------------------------------------------------
    # checkpointing
    # --------------------------------------------------
    def save(self, path, step=None):
        obj = {
            "model": self.imagen.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step if step is None else step
        }
        torch.save(obj, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.imagen.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        # Move model to the correct device after loading
        self.imagen.to(self.device)
