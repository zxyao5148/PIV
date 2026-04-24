"""
Train PIV and baseline models (ZNet, AutoIV, VIV, GIV, DIV.VAE) on the
geom_v benchmark and save a combined diagnostic figure.

Usage examples:
    python scripts/run_geom_v.py                          # all geometries, all methods
    python scripts/run_geom_v.py --methods PIV ZNet        # only PIV and ZNet
    python scripts/run_geom_v.py --geometries spiral gmm   # only two geometries
    python scripts/run_geom_v.py --methods PIV --geometries pinwheel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.sim_geom_v import GEOMETRY_SAMPLERS, GeomVConfig, simulate_geom_v
from synthetic_dataset import make_synthetic_proxyiv_dataloaders
from trainer import (
    pretrain_znet_xty,
    train_autoiv,
    train_baseline,
    train_giv,
    train_stage1,
    train_stage2,
)
from evaluation import run_probe
from utils import collect_loader, extract_cz, extract_wz, seed_everything
from vis.vis_geom_v import make_geom_v_figure

from models.proxyiv import build_proxyiv
from models.znet import build_znet
from models.autoiv import build_autoiv
from models.viv import build_viv
from models.giv import build_giv
from models.divvae import build_divvae

# ── Dimensions (fixed for this benchmark) ─────────────────────────
X_DIM = 4   # X ∈ R⁴ from diffeomorphic_mix(C, V, D)
M_DIM = 1   # scalar proxy
W_DIM = 1   # C is scalar  → 1-D confounding representation
Z_DIM = 2   # V is 2-D     → 2-D instrument representation
A_DIM = 2   # 2-D treatment
L_DIM = 0   # no observed covariates L
Y_DIM = 1   # scalar outcome
C_DIM = W_DIM  # baselines use the same budget for confounder repr

ALL_METHODS = ["PIV", "ZNet", "AutoIV", "VIV", "GIV", "DIV.VAE"]
ALL_GEOMETRIES = list(GEOMETRY_SAMPLERS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geom_v benchmark: PIV + baselines")
    p.add_argument("--n", type=int, default=8_000, help="samples per geometry")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--s1_epochs", type=int, default=100, help="PIV stage-1 epochs")
    p.add_argument("--s2_epochs", type=int, default=100, help="PIV stage-2 epochs")
    p.add_argument("--epochs", type=int, default=200, help="training epochs (ZNet/VIV/DIV.VAE/AutoIV)")
    p.add_argument("--probe_epochs", type=int, default=200, help="(C,Z)->V probe epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        choices=ALL_METHODS, metavar="METHOD",
        help=f"methods to run (default: all). Choices: {', '.join(ALL_METHODS)}",
    )
    p.add_argument(
        "--geometries", nargs="+", default=ALL_GEOMETRIES,
        choices=ALL_GEOMETRIES, metavar="GEOM",
        help=f"geometries to run (default: all). Choices: {', '.join(ALL_GEOMETRIES)}",
    )
    return p.parse_args()


# ── Per-method runners ────────────────────────────────────────────
# Each returns V_recon numpy array for the test set.

def _run_piv(loaders, args, device):
    piv = build_proxyiv(
        x_dim=X_DIM, m_dim=M_DIM, l_dim=L_DIM,
        w_dim=W_DIM, z_dim=Z_DIM, a_dim=A_DIM,
        treatment_family="continuous",
    ).to(device)

    train_stage1(piv, loaders.train, args.s1_epochs, args.lr, device)
    piv.copy_stage1_into_stage2(freeze=True)
    train_stage2(piv, loaders.train, args.s2_epochs, args.lr, device)

    w_tr, z_tr, v_tr = extract_wz(piv, loaders.train, device, target_key="V")
    w_te, z_te, v_te = extract_wz(piv, loaders.test, device, target_key="V")
    return run_probe(w_tr, z_tr, v_tr, w_te, z_te, v_te,
                     epochs=args.probe_epochs, lr=args.lr, device=device).numpy()


def _run_znet(loaders, args, device):
    znet = build_znet(
        x_dim=X_DIM, c_dim=C_DIM, z_dim=Z_DIM, a_dim=A_DIM,
        y_dim=Y_DIM, treatment_family="continuous",
    ).to(device)

    pretrain_znet_xty(znet, loaders.train, device, epochs=200, lr=args.lr)
    train_baseline(znet, loaders.train, args.epochs, args.lr, device, label="ZNet")

    c_tr, z_tr, v_tr = extract_cz(znet, loaders.train, device, target_key="V")
    c_te, z_te, v_te = extract_cz(znet, loaders.test, device, target_key="V")
    return run_probe(c_tr, z_tr, v_tr, c_te, z_te, v_te,
                     epochs=args.probe_epochs, lr=args.lr, device=device).numpy()


def _run_autoiv(loaders, args, device):
    autoiv = build_autoiv(
        x_dim=X_DIM, rep_dim=Z_DIM, a_dim=A_DIM,
        y_dim=Y_DIM, treatment_family="continuous",
    ).to(device)

    train_autoiv(autoiv, loaders.train, args.epochs, args.lr, device)

    c_tr, z_tr, v_tr = extract_cz(autoiv, loaders.train, device, target_key="V")
    c_te, z_te, v_te = extract_cz(autoiv, loaders.test, device, target_key="V")
    return run_probe(c_tr, z_tr, v_tr, c_te, z_te, v_te,
                     epochs=args.probe_epochs, lr=args.lr, device=device).numpy()


def _run_viv(loaders, args, device):
    viv = build_viv(
        x_dim=X_DIM, c_dim=C_DIM, z_dim=Z_DIM, a_dim=A_DIM,
        y_dim=Y_DIM, treatment_family="continuous",
    ).to(device)

    train_baseline(viv, loaders.train, args.epochs, args.lr, device, label="VIV")

    c_tr, z_tr, v_tr = extract_cz(viv, loaders.train, device, target_key="V")
    c_te, z_te, v_te = extract_cz(viv, loaders.test, device, target_key="V")
    return run_probe(c_tr, z_tr, v_tr, c_te, z_te, v_te,
                     epochs=args.probe_epochs, lr=args.lr, device=device).numpy()


def _run_giv(loaders, args, device):
    giv = build_giv(
        x_dim=X_DIM, rep_dim=X_DIM, a_dim=A_DIM,
        n_domains=3, treatment_family="continuous",
    ).to(device)

    train_giv(giv, loaders.train, device, em_iters=20, inner_epochs=200, lr=args.lr)

    c_tr, z_tr, v_tr = extract_cz(giv, loaders.train, device, target_key="V")
    c_te, z_te, v_te = extract_cz(giv, loaders.test, device, target_key="V")
    return run_probe(c_tr, z_tr, v_tr, c_te, z_te, v_te,
                     epochs=args.probe_epochs, lr=args.lr, device=device).numpy()


def _run_divvae(loaders, args, device):
    divvae = build_divvae(
        x_dim=X_DIM, c_dim=C_DIM, z_dim=Z_DIM, a_dim=A_DIM,
        y_dim=Y_DIM, treatment_family="continuous",
    ).to(device)

    train_baseline(divvae, loaders.train, args.epochs, args.lr, device, label="DIV.VAE")

    c_tr, z_tr, v_tr = extract_cz(divvae, loaders.train, device, target_key="V")
    c_te, z_te, v_te = extract_cz(divvae, loaders.test, device, target_key="V")
    return run_probe(c_tr, z_tr, v_tr, c_te, z_te, v_te,
                     epochs=args.probe_epochs, lr=args.lr, device=device).numpy()


METHOD_RUNNERS = {
    "PIV": _run_piv,
    "ZNet": _run_znet,
    "AutoIV": _run_autoiv,
    "VIV": _run_viv,
    "GIV": _run_giv,
    "DIV.VAE": _run_divvae,
}


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    methods = list(args.methods)
    geometries = list(args.geometries)

    print(f"Geometries : {', '.join(geometries)}")
    print(f"Methods    : {', '.join(methods)}")

    vis_data: dict[str, dict] = {}

    for geom in geometries:
        print(f"\n{'=' * 60}")
        print(f"  Geometry: {geom}")
        print(f"{'=' * 60}")

        cfg = GeomVConfig(n=args.n, geometry=geom, seed=args.seed)
        data = simulate_geom_v(cfg)

        loaders = make_synthetic_proxyiv_dataloaders(
            arrays=data,
            keys=["X", "M", "A", "Y", "V"],
            batch_size=args.batch_size,
            seed=args.seed,
        )

        geom_vis: dict[str, object] = {}

        test_arrays = collect_loader(loaders.test, keys=["V"])
        geom_vis["V_true"] = test_arrays["V"].numpy()

        for method in methods:
            print(f"\n[{method}] Training ...")
            geom_vis[f"V_{method}"] = METHOD_RUNNERS[method](loaders, args, device)

        vis_data[geom] = geom_vis

    # ── Save combined figure ─────────────────────────────────────
    out_dir = PROJECT_ROOT / "outputs" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = make_geom_v_figure(rows=geometries, data=vis_data, methods=methods)
    fig_path = out_dir / "geom_v.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    main()
