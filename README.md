- First install

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- init dvc: `dvc init`
- use the standard config to push and pull from local file system:

```bash
dvc add data/asset.txt
git commit -m "added file to local storage"
dvc push

```
