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

- if you have a storage on azure instead:

```bash
dvc remote add blob azure://mycontainer/myfolder  --local
```

then in the .dvc/.config.local file:

```bash
['remote "blob"']
    url = azure://mycontainer/myfolder
    connection_string = <my_connection_string>
[core]
    remote = blob
```

config.local is not committed to source control and is used in priority

If you don't want to use a connection string, create a service principal (https://learn.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal) and assign it the role Storage Blob Data Contributor in the IAM of the storage account, then in the config

```bash
['remote "blob"']
    url = azure://mycontainer/myfolder
    account_name = storageaccountname
    tenant_id = <tenant-id>
    client_id = <client-id>
    client_secret = <client-secret> # must be created for the SP
[core]
    remote = blob
```
