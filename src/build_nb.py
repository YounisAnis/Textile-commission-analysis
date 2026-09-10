"""Small helper: turn a list of (kind, text) blocks into an executed notebook."""
import sys
import nbformat as nbf
from nbclient import NotebookClient


def build(blocks, out_path, timeout=900):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t)
        for k, t in blocks
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": "notebooks/"}})
    client.execute()
    nbf.write(nb, out_path)
    errors = [
        (i, o.get("evalue"))
        for i, c in enumerate(nb.cells)
        for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    if errors:
        print("ERRORS:", errors, file=sys.stderr)
    else:
        print(f"OK -> {out_path}")
    return len(errors) == 0
