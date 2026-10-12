import json
import tomllib
from pathlib import Path

CONFIG = tomllib.loads(Path(__file__).with_name("config.toml").read_text())


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)
