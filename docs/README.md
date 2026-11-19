# Documentation

- `index.html` — the get-started page (Bluetooth tool), published to [pollen-robotics.github.io/grabette](https://pollen-robotics.github.io/grabette/) from `main`.

## Preview the site locally

From the repository root (needs Node and npm):

```bash
uvx --from git+https://github.com/huggingface/doc-builder@main --with watchdog \
  doc-builder preview grabette docs/source --not_python_module
```

Open the printed URL followed by `/index`, e.g. <http://localhost:5173/index>; the bare `/` is a 404 in the preview. Pages reload when you edit `source/`.

Use doc-builder from GitHub `main`, as above: the PyPI release fails to serve the pages.
