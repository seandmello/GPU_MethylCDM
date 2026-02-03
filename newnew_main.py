import argparse
import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision import utils as vutils
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from new_read_data import PatchRNADataset
from RNA_cdm import RNACDM
from unet import Unet
from new_trainer import RNACDMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MethylCDM")
    parser.add_argument("--path_to_patches", type=str, required=True,
                        help="Path to directory containing WSI patch folders")
    parser.add_argument("--path_to_methyl", type=str, required=True,
                        help="Path to directory containing per-WSI .npy methylation vectors")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batch_size", type=int, default=128,
                        help="Max sub-batch size for gradient accumulation chunking")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=256,
                        help="Base dimension of the UNet")
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Dimension multipliers for each UNet level")
    parser.add_argument("--num_resnet_blocks", type=int, default=3,
                        help="Number of ResNet blocks per level")
    parser.add_argument("--layer_attns", type=int, nargs="+", default=[0, 1, 1, 1],
                        help="Self-attention per level (0=False, 1=True)")
    parser.add_argument("--layer_cross_attns", type=int, nargs="+", default=[0, 1, 1, 1],
                        help="Cross-attention per level (0=False, 1=True)")
    parser.add_argument("--attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--ff_mult", type=float, default=2.0,
                        help="Feed-forward multiplier")
    parser.add_argument("--memory_efficient", action="store_true", default=False,
                        help="Use memory-efficient attention")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (only used when CUDA is available)")
    parser.add_argument("--num_iter_save", type=int, default=500,
                        help="Save a checkpoint every N training steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    methyl = True

    training_dataset = PatchRNADataset(args.path_to_patches, args.path_to_methyl)
    print(f"Dataset size: {len(training_dataset)} samples")

    unet1 = Unet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        num_resnet_blocks=args.num_resnet_blocks,
        layer_attns=tuple(bool(x) for x in args.layer_attns),
        layer_cross_attns=tuple(bool(x) for x in args.layer_cross_attns),
        attn_heads=args.attn_heads,
        ff_mult=args.ff_mult,
        memory_efficient=args.memory_efficient,
        cond_dim=10 if methyl else 0,
        cond_on_rna=methyl,
        max_rna_len=10 if methyl else 0,
    )

    imagen = RNACDM(
        unets=(unet1,),
        image_sizes=(64,),
        timesteps=args.timesteps,
        cond_drop_prob=0.5,
        condition_on_rna=methyl,
        rna_embed_dim=10 if methyl else 0,
    )

    trainer = RNACDMTrainer(imagen, lr=args.lr)

    step = 0
    start_epoch = 0

    if args.resume is not None:
        trainer.load(args.resume)
        step = trainer.global_step
        print(f"Resumed from checkpoint: {args.resume}  (step {step})")

    # --- Adaptive DataLoader (CPU vs GPU) ---
    use_cuda = torch.cuda.is_available()
    num_workers = args.num_workers if use_cuda else 0
    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )
    train_dl = torch.utils.data.DataLoader(training_dataset, **dl_kwargs)

    device_name = "cuda" if use_cuda else "cpu"
    print(f"Device: {device_name}")
    print(f"DataLoader: batch_size={args.batch_size}, num_workers={num_workers}, pin_memory={use_cuda}")
    print("Starting training")

    os.makedirs(args.save_dir, exist_ok=True)

    # Loss tracking
    step_losses = []       # (step, loss) for every logged step
    epoch_avg_losses = []  # average loss per epoch

    for epoch in range(start_epoch, args.num_epochs):
        epoch_losses = []
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            images = batch["image"]
            methyl_data = batch["methyl_data"] if methyl else None

            loss = trainer(
                images,
                methyl_embeds=methyl_data,
                unet_number=1,
                max_batch_size=args.max_batch_size,
            )
            trainer.update(unet_number=1)
            step += 1
            epoch_losses.append(loss)

            if step % 50 == 0:
                print(f"  epoch={epoch+1} step={step} loss={loss:.4f}")
                step_losses.append((step, loss))

            if trainer.is_main and step % args.num_iter_save == 0:
                ckpt_path = os.path.join(args.save_dir, f"model-step{step}.pt")
                trainer.save(ckpt_path, step)
                print(f"  Saved checkpoint: {ckpt_path}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(avg_loss)
        print(f"  Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Save plots at the end of each epoch
        plot_dir = os.path.join(args.save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Step-level loss plot
        if step_losses:
            fig, ax = plt.subplots(figsize=(10, 5))
            steps_x, losses_y = zip(*step_losses)
            ax.plot(steps_x, losses_y, linewidth=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss (per step)")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "loss_per_step.png"), dpi=150)
            plt.close(fig)

        # Epoch-level loss plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(epoch_avg_losses) + 1), epoch_avg_losses, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Loss")
        ax.set_title("Training Loss (per epoch)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "loss_per_epoch.png"), dpi=150)
        plt.close(fig)

        # Generate 5 sample patches at the end of each epoch
        if trainer.is_main and methyl:
            gen_dir = os.path.join(args.save_dir, "samples", f"epoch_{epoch+1}")
            os.makedirs(gen_dir, exist_ok=True)

            device = next(imagen.parameters()).device
            rna_files = sorted([f for f in os.listdir(args.path_to_methyl) if f.endswith(".npy")])
            if rna_files:
                rna_vector = np.load(os.path.join(args.path_to_methyl, rna_files[0]))
                rna_embed = torch.from_numpy(rna_vector).float().unsqueeze(0).to(device)
                rna_batch = rna_embed.expand(5, -1)

                with torch.no_grad():
                    generated = trainer.sample(
                        batch_size=5,
                        rna_embeds=rna_batch,
                        cond_scale=3.0,
                        stop_at_unet_number=1,
                        return_pil_images=False,
                    )

                for i in range(generated.shape[0]):
                    vutils.save_image(generated[i], os.path.join(gen_dir, f"tile_{i}.png"))

                grid = vutils.make_grid(generated, nrow=5, normalize=True)
                vutils.save_image(grid, os.path.join(gen_dir, "grid.png"))
                print(f"  Saved 5 sample patches to {gen_dir}")

    if trainer.is_main:
        final_path = os.path.join(args.save_dir, "model-final.pt")
        trainer.save(final_path, step)
        print(f"Training complete. Final model saved to {final_path}")
        print(f"Loss plots saved to {os.path.join(args.save_dir, 'plots')}")
