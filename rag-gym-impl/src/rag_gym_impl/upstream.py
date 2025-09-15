import os
import types
import importlib.util
from functools import lru_cache


def get_project_root() -> str:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(package_dir, "../../.."))


def get_upstream_root() -> str:
    return os.path.join(get_project_root(), "RAG-Gym")


def _resolve_upstream_file_path(dotted_module_path: str) -> str:
    # Example: "rag_gym.envs.state" -> "<UPSTREAM>/rag_gym/envs/state.py"
    relative_path = dotted_module_path.replace(".", os.sep) + ".py"
    file_path = os.path.join(get_upstream_root(), relative_path)
    return file_path


@lru_cache(maxsize=None)
def load_upstream_module(dotted_module_path: str) -> types.ModuleType:
    file_path = _resolve_upstream_file_path(dotted_module_path)
    if not os.path.isfile(file_path):
        raise ImportError(
            f"Upstream module not found at: {file_path}. "
            f"Expected to load '{dotted_module_path}' from RAG-Gym submodule."
        )
    spec = importlib.util.spec_from_file_location(dotted_module_path, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for upstream module '{dotted_module_path}' at '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def import_symbol(dotted_module_path: str, symbol_name: str):
    module = load_upstream_module(dotted_module_path)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(
            f"Symbol '{symbol_name}' not found in upstream module '{dotted_module_path}'."
        ) from exc


def import_symbols(dotted_module_path: str, *symbol_names: str):
    module = load_upstream_module(dotted_module_path)
    resolved = []
    for name in symbol_names:
        if not hasattr(module, name):
            raise ImportError(
                f"Symbol '{name}' not found in upstream module '{dotted_module_path}'."
            )
        resolved.append(getattr(module, name))
    return tuple(resolved)


