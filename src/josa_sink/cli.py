import argparse
from .config import load_config
from .pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
