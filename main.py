"""RL Box World — entry point.

Usage examples:
  python main.py                            # headless training, 1000 episodes
  python main.py --render                   # training with pygame window
  python main.py --episodes 3000            # more episodes
  python main.py --render --no-train --checkpoint checkpoints/dqn_final.pt
"""

import argparse

from src.config import Config
from src.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL Box World")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open pygame window while training.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        dest="no_train",
        help="Disable learning (watch a saved model).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to run (default: 1000).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved .pt checkpoint to resume from.",
    )
    # optional world overrides
    parser.add_argument("--grid-w",   type=int, default=None, help="Grid width.")
    parser.add_argument("--grid-h",   type=int, default=None, help="Grid height.")
    parser.add_argument("--agents",   type=int, default=None, help="Number of agents.")
    parser.add_argument("--apples",   type=int, default=None, help="Number of apples.")
    parser.add_argument("--lr",       type=float, default=None, help="Learning rate.")
    parser.add_argument("--fps",      type=int,   default=None, help="Renderer FPS cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    if args.grid_w   is not None: cfg.grid_width  = args.grid_w
    if args.grid_h   is not None: cfg.grid_height = args.grid_h
    if args.agents   is not None: cfg.num_agents  = args.agents
    if args.apples   is not None: cfg.num_apples  = args.apples
    if args.lr       is not None: cfg.lr          = args.lr
    if args.fps      is not None: cfg.fps         = args.fps

    print("=== RL Box World ===")
    print(f"  Grid       : {cfg.grid_width} x {cfg.grid_height}")
    print(f"  Agents     : {cfg.num_agents}")
    print(f"  Apples     : {cfg.num_apples}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Render     : {args.render}")
    print(f"  No-train   : {args.no_train}")
    print(f"  Checkpoint : {args.checkpoint}")
    print()

    run_training(
        cfg=cfg,
        num_episodes=args.episodes,
        render=args.render,
        no_train=args.no_train,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
