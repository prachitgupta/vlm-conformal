import ast
from pathlib import Path


def _load_console_scripts(setup_py: Path) -> set[str]:
    tree = ast.parse(setup_py.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != "entry_points" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if not isinstance(key, ast.Constant) or key.value != "console_scripts":
                    continue
                return {
                    entry.value.split(" = ", 1)[0]
                    for entry in value.elts
                    if isinstance(entry, ast.Constant) and isinstance(entry.value, str)
                }
    raise AssertionError("setup.py is missing a console_scripts entry_points block")


def test_console_scripts_match_wrapper_files():
    package_root = Path(__file__).resolve().parents[1]
    console_scripts = _load_console_scripts(package_root / "setup.py")
    wrapper_files = {
        path.name for path in (package_root / "scripts").iterdir()
        if path.is_file() and path.name != "__init__.py" and "." not in path.name
    }

    assert wrapper_files == console_scripts
