import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """\
    # Automatic Juputer Notebook.
    This Notebook is auto-generated."""

code = """\
    %matplotlib inline
    import numpy as np
    import pandas as pd
    import pandas_profiling
    import matplotlib.pyplot as plt
    import seaborn as sns

    """

nb['cells'] = [nbf.v4.new_markdown_cell(text),
                nbf.v4.new_code_cell(code)]

nbf.write(nb, 'test.ipynb')
    