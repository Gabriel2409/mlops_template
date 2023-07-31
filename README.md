# Mlops template

Heavily inspired by https://madewithml.com/


# Kedro
https://github.com/kedro-org/kedro/issues/1271



- First install

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## DVC

- init dvc: `dvc init`

### Local file system

- use the standard config to push and pull from local file system:

```bash
dvc add data/asset.txt
git commit -m "added file to local storage"
dvc push
```

### Azure blob storage

- you can manually deploy the azure infrastructure with `az deployment group create --resource-group mlops_exps --template-file infrastructure/main.bicep --parameters infrastructure/default_parameters.bicepparam`

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

Note that you can get the connection string: `az storage account show-connection-string --name mlopsstorage124fxgc --key key1`

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
