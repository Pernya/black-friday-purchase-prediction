"""Static artifact checks; does not train models or collect external data."""
from pathlib import Path
import ast
import nbformat
from IPython.core.inputtransformer2 import TransformerManager

root = Path(__file__).resolve().parents[1]
transformer = TransformerManager()
notebooks = list(root.glob("*.ipynb"))
assert notebooks, "No project notebook found"
cells = 0
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(transformer.transform_cell(cell.source), f"{path.name}:cell-{i}", "exec")
            cells += 1
for path in root.rglob("*.py"):
    if not any(part.startswith(".") for part in path.relative_to(root).parts):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(root)))
print(f"OK: {len(notebooks)} notebook(s), {cells} code cells; Python syntax validated.")
print("No model training, live collection, or end-to-end notebook execution performed.")
