import json
from pathlib import Path
from typing import List, Dict, Any


def load_minpairs_suite(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_abstention_suite(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))