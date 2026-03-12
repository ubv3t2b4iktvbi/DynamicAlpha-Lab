# notebooks

Notebook families are grouped by research theme so future demos stay discoverable.

## Layout

- `sf/`: slow-fast demos, ablations, and theory notebooks

## Naming

Use:

`<scope>_<family>_demo.ipynb`

Guidelines:

- `<scope>` is the system name, observation regime, or suite tag
- `<family>` is the short research family tag such as `sf`, `factor`, or `koopman`
- keep paired outputs under `runs/demo_notebook/<family>/<scope>/`

Examples:

- `hindmarsh_sf_demo.ipynb`
- `lorenz96_sparse_sf_demo.ipynb`
- `classic_sparse_sf_demo.ipynb`

The same convention can be reused for non-slow-fast factor notebooks later.
