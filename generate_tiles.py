import argparse
import os
import torch
import numpy as np
from torchvision import utils as vutils

from RNA_cdm import RNACDM
from unet import Unet
from new_trainer import RNACDMTrainer

methyl = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tiles from a pre-trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--rna_dir', type=str, required=True,
                        help='Directory containing per-WSI .npy RNA/methylation vectors')
    parser.add_argument('--save_dir', type=str, default='generated_images/',
                        help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to generate per WSI')
    parser.add_argument('--cond_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Model architecture args (must match training config)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=256,
                        help='Base dimension of the UNet')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Dimension multipliers for each UNet level')
    parser.add_argument('--num_resnet_blocks', type=int, default=3,
                        help='Number of ResNet blocks per level')
    parser.add_argument('--layer_attns', type=int, nargs='+', default=[0, 1, 1, 1],
                        help='Self-attention per level (0=False, 1=True)')
    parser.add_argument('--layer_cross_attns', type=int, nargs='+', default=[0, 1, 1, 1],
                        help='Cross-attention per level (0=False, 1=True)')
    parser.add_argument('--attn_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--ff_mult', type=float, default=2.0,
                        help='Feed-forward multiplier')
    parser.add_argument('--memory_efficient', action='store_true', default=False,
                        help='Use memory-efficient attention')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the same model architecture used during training
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
        max_rna_len=10 if methyl else 0
    )

    imagen = RNACDM(
        unets=(unet1,),
        image_sizes=(64,),
        timesteps=args.timesteps,
        cond_drop_prob=0.5,
        condition_on_rna=methyl,
        rna_embed_dim=10 if methyl else 0
    )

    trainer = RNACDMTrainer(imagen, lr=args.lr)
    trainer.load(args.checkpoint)
    print(f'Loaded checkpoint from {args.checkpoint}')

    os.makedirs(args.save_dir, exist_ok=True)

    # Collect all WSI RNA vectors
    rna_files = sorted([f for f in os.listdir(args.rna_dir) if f.endswith('.npy')])
    print(f'Found {len(rna_files)} WSI RNA vectors')

    for rna_file in rna_files:
        wsi_name = os.path.splitext(rna_file)[0]
        rna_vector = np.load(os.path.join(args.rna_dir, rna_file))
        rna_embed = torch.from_numpy(rna_vector).float().unsqueeze(0).to(device)

        # Repeat the conditioning vector for the batch
        rna_batch = rna_embed.expand(args.num_images, -1)

        print(f'Generating {args.num_images} images for {wsi_name}...')
        generated = trainer.sample(
            batch_size=args.num_images,
            rna_embeds=rna_batch,
            cond_scale=args.cond_scale,
            stop_at_unet_number=1,
            return_pil_images=False
        )

        # Save each image individually
        wsi_dir = os.path.join(args.save_dir, wsi_name)
        os.makedirs(wsi_dir, exist_ok=True)

        for i in range(generated.shape[0]):
            img_path = os.path.join(wsi_dir, f'tile_{i}.png')
            vutils.save_image(generated[i], img_path)

        # Also save a grid view
        grid = vutils.make_grid(generated, nrow=min(args.num_images, 8), normalize=True)
        vutils.save_image(grid, os.path.join(wsi_dir, 'grid.png'))

    print(f'Done. Images saved to {args.save_dir}')
