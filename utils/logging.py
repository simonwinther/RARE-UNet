#utils/logging.py

import logging, os, sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


DEFAULT_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_is_logging_configured = False

# module‐level guard
def setup_logging(logging_cfg: DictConfig, exp_dir: Path):
    if logging_cfg is None:
        raise ValueError("Logging configuration is required.")

    if logging_cfg.get("disable_all", True):
        print("Logging is disabled by configuration.")
        logging.disable(logging.CRITICAL)
        return

    global _is_logging_configured
    if _is_logging_configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    root.propagate = False

    # ---- If we don't have this, we will get default logging that logging provides us with, we dont :)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    file_cfg = logging_cfg.get("file", {})
    console_cfg = logging_cfg.get("console", {})

    file_enabled = file_cfg.get("level") is not None
    console_enabled = console_cfg.get("level") is not None

    if file_enabled:
        lvl = getattr(logging, file_cfg["level"].upper(), logging.INFO)
        fh = logging.FileHandler(os.path.join(exp_dir, "output.log"))
        fh.setLevel(lvl)
        fh.setFormatter(
            logging.Formatter(
                file_cfg.get("format", DEFAULT_FMT),
                datefmt=file_cfg.get("datefmt", DEFAULT_DATEFMT),
            )
        )

        root.addHandler(fh)
        root.debug(f"File logging enabled at {file_cfg['level']}")

    if console_enabled:
        lvl = getattr(logging, console_cfg["level"].upper(), logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(
            logging.Formatter(
                console_cfg.get("format", DEFAULT_FMT),
                datefmt=console_cfg.get("datefmt", DEFAULT_DATEFMT),
            )
        )
        root.addHandler(ch)
        root.debug(f"Console logging enabled at {console_cfg['level']}")

    _is_logging_configured = True
    root.debug("Root logger configuration complete.")
