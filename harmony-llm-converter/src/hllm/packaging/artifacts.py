from __future__ import annotations

import shutil
from pathlib import Path


def assemble_artifacts(source: Path, final_dir: Path) -> list[tuple[Path, str]]:
    """Copy runtime resources into the final layout and return package entries."""
    final_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    config_names = {"generation_config.json", "chat_template.json"}

    for name in sorted(tokenizer_names | config_names):
        src = source / name
        if not src.is_file():
            continue
        bucket = "tokenizer" if name in tokenizer_names else "config"
        destination = final_dir / bucket / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)

    config_file = source / "config.json"
    if config_file.is_file():
        destination = final_dir / "config" / "model.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_file, destination)

    return [
        (path, f"model/{path.relative_to(final_dir).as_posix()}")
        for path in sorted(final_dir.rglob("*"))
        if path.is_file()
    ]
