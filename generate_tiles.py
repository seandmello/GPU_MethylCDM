import argparse
import json
import os
import torch
import numpy as np
from torchvision import utils as vutils

from RNA_cdm import RNACDM
from unet import Unet
from new_trainer import RNACDMTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tiles from a pre-trained model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model_config.json saved by training')
    parser.add_argument('--cancer_type', type=str, default=None,
                        help='Cancer type to generate (e.g. TCGA-BLCA). Uses cancer_types from config.')
    parser.add_argument('--rna_dir', type=str, default=None,
                        help='Directory containing per-WSI .npy RNA/methylation vectors')
    parser.add_argument('--save_dir', type=str, default='generated_images/',
                        help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to generate')
    parser.add_argument('--cond_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Load model config from training
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    methyl = cfg.get('condition_on_rna', False)
    cond_dim = cfg.get('cond_dim', 10 if methyl else 0)
    cancer_types = cfg.get('cancer_types', None)

    print(f"Loaded model config from {args.config}")
    print(f"  checkpoint      : {cfg['checkpoint']}")
    print(f"  condition_on_rna: {methyl}")
    print(f"  cond_dim        : {cond_dim}")
    if cancer_types:
        print(f"  cancer_types    : {cancer_types}")
    print(f"  dim             : {cfg['dim']}")
    print(f"  dim_mults       : {cfg['dim_mults']}")
    print(f"  timesteps       : {cfg['timesteps']}")

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model from config
    unet1 = Unet(
        dim=cfg['dim'],
        dim_mults=tuple(cfg['dim_mults']),
        num_resnet_blocks=cfg['num_resnet_blocks'],
        layer_attns=tuple(bool(x) for x in cfg['layer_attns']),
        layer_cross_attns=tuple(bool(x) for x in cfg['layer_cross_attns']),
        attn_heads=cfg['attn_heads'],
        ff_mult=cfg['ff_mult'],
        memory_efficient=cfg.get('memory_efficient', False),
        cond_dim=cond_dim if methyl else 0,
        cond_on_rna=methyl,
        max_rna_len=cond_dim if methyl else 0
    )

    imagen = RNACDM(
        unets=(unet1,),
        image_sizes=(64,),
        timesteps=cfg['timesteps'],
        cond_drop_prob=0.5,
        condition_on_rna=methyl,
        rna_embed_dim=cond_dim if methyl else 0
    )

    trainer = RNACDMTrainer(imagen, lr=cfg['lr'])
    trainer.load(cfg['checkpoint'])
    print(f'Loaded checkpoint from {cfg["checkpoint"]}')

    os.makedirs(args.save_dir, exist_ok=True)

    if cancer_types and args.cancer_type:
        # --- Cancer-type conditioned generation ---
        assert args.cancer_type in cancer_types, \
            f"Unknown cancer type '{args.cancer_type}'. Available: {cancer_types}"

        ct_idx = cancer_types.index(args.cancer_type)
        one_hot = torch.zeros(1, cond_dim, device=device)
        one_hot[0, ct_idx] = 1.0
        rna_batch = one_hot.expand(args.num_images, -1)

        print(f'Generating {args.num_images} images for {args.cancer_type} (index {ct_idx})...')
        generated = trainer.sample(
            batch_size=args.num_images,
            rna_embeds=rna_batch,
            cond_scale=args.cond_scale,
            stop_at_unet_number=1,
            return_pil_images=False
        )

        ct_dir = os.path.join(args.save_dir, args.cancer_type)
        os.makedirs(ct_dir, exist_ok=True)
        for i in range(generated.shape[0]):
            vutils.save_image(generated[i], os.path.join(ct_dir, f'tile_{i}.png'))

        grid = vutils.make_grid(generated, nrow=min(args.num_images, 8), normalize=True)
        vutils.save_image(grid, os.path.join(ct_dir, 'grid.png'))

    elif cancer_types and args.cancer_type is None:
        # Generate for all cancer types
        for ct_idx, ct_name in enumerate(cancer_types):
            one_hot = torch.zeros(1, cond_dim, device=device)
            one_hot[0, ct_idx] = 1.0
            rna_batch = one_hot.expand(args.num_images, -1)

            print(f'Generating {args.num_images} images for {ct_name} (index {ct_idx})...')
            generated = trainer.sample(
                batch_size=args.num_images,
                rna_embeds=rna_batch,
                cond_scale=args.cond_scale,
                stop_at_unet_number=1,
                return_pil_images=False
            )

            ct_dir = os.path.join(args.save_dir, ct_name)
            os.makedirs(ct_dir, exist_ok=True)
            for i in range(generated.shape[0]):
                vutils.save_image(generated[i], os.path.join(ct_dir, f'tile_{i}.png'))

            grid = vutils.make_grid(generated, nrow=min(args.num_images, 8), normalize=True)
            vutils.save_image(grid, os.path.join(ct_dir, 'grid.png'))

    elif methyl and args.rna_dir:
        # --- RNA/methylation conditioned generation ---
        rna_files = sorted([f for f in os.listdir(args.rna_dir) if f.endswith('.npy')])
        print(f'Found {len(rna_files)} WSI RNA vectors')

        for rna_file in rna_files:
            wsi_name = os.path.splitext(rna_file)[0]
            rna_vector = np.load(os.path.join(args.rna_dir, rna_file))
            rna_embed = torch.from_numpy(rna_vector).float().unsqueeze(0).to(device)
            rna_batch = rna_embed.expand(args.num_images, -1)

            print(f'Generating {args.num_images} images for {wsi_name}...')
            generated = trainer.sample(
                batch_size=args.num_images,
                rna_embeds=rna_batch,
                cond_scale=args.cond_scale,
                stop_at_unet_number=1,
                return_pil_images=False
            )

            wsi_dir = os.path.join(args.save_dir, wsi_name)
            os.makedirs(wsi_dir, exist_ok=True)
            for i in range(generated.shape[0]):
                vutils.save_image(generated[i], os.path.join(wsi_dir, f'tile_{i}.png'))

            grid = vutils.make_grid(generated, nrow=min(args.num_images, 8), normalize=True)
            vutils.save_image(grid, os.path.join(wsi_dir, 'grid.png'))

    else:
        # --- Unconditional generation ---
        print(f'Generating {args.num_images} unconditional images...')
        generated = trainer.sample(
            batch_size=args.num_images,
            stop_at_unet_number=1,
            return_pil_images=False
        )

        for i in range(generated.shape[0]):
            vutils.save_image(generated[i], os.path.join(args.save_dir, f'tile_{i}.png'))

        grid = vutils.make_grid(generated, nrow=min(args.num_images, 8), normalize=True)
        vutils.save_image(grid, os.path.join(args.save_dir, 'grid.png'))

    print(f'Done. Images saved to {args.save_dir}')
