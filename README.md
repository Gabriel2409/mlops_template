# Mlops template

Heavily inspired by https://madewithml.com/


# Kedro
https://github.com/kedro-org/kedro/issues/1271

Run kedro mlflow init and in the mlflow.yaml, change traching uri to point to azure ml
workspace if you want to develop locally. If you develop directly on aml workspace,
no need to change tracking uri


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


# github action to push arm template
https://learn.microsoft.com/en-us/azure/azure-resource-manager/templates/deploy-github-actions?tabs=userlevel

- get subscription and rg id: `az group list`
- create a contributor service principal:
```bash
az ad sp create-for-rbac \
--name "<myrgcontributorsp>" --role contributor \
--scopes /subscriptions/<subscription-id>/resourceGroups/<group-name> \
--sdk-auth
```
- add full json object to github repository secrets under name AZURE_CREDENTIALS
- also create other secrets: AZURE_SUBSCRIPTION and AZURE_RG and AZURE_LOCATION
- add the repository variable BASE_NAME

# azure devops to push arms template

- no need to create a sp manually, instead go to project settings > service connections
then choose Azure Resource manager and configure the SP here

- Create a new variable group named mlopsvars (Pipelines / Library) and you can easily add variables.
- Add AZURE_RG, AZURE_LOCATION, BASE_NAME and AZURE_RM_SVC_CONNECTION (the name of the connector that was just created)
- Click on the lock to make the variables secrets
