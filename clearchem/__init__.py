from .clearchem import ClearChem
__all__ = ["ClearChem", "ChemQwen"]
__version__ = "1.0.0"


def __getattr__(name):
    # 知识层按需导入：它要加载 54GB 底座，不该在 import clearchem 时就触发
    if name == "ChemQwen":
        from .knowledge import ChemQwen
        return ChemQwen
    raise AttributeError(name)
