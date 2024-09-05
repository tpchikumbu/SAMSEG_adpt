import nbformat as nbf
import os

dirs = os.walk('.')
# Find all the python files in the list of directories
files = []
for d in dirs:
    for f in d[2]:
        if f.endswith('.py'):
            # Add the full filepath to the list
            files.append(os.path.join(d[0], f))

# Create a new notebook
nb = nbf.v4.new_notebook()

for file in files:
    with open(file, 'r') as f:
        code = f.read()
    # Add filename as a markdown cell
    nb.cells.append(nbf.v4.new_markdown_cell('# ' + file))

    # Add each file's code to a new cell
    nb.cells.append(nbf.v4.new_code_cell(code))

# Save the notebook
with open('combined_notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
