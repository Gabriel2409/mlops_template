# Mlops template

Heavily inspired by https://madewithml.com/

## First install

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Provision dev resources on azure

### Manual Deployment

- you can manually deploy the azure infrastructure with
  `az deployment group create --resource-group <myrg> --template-file infrastructure/main.bicep --parameters infrastructure/default_parameters.bicepparam`

### Deployment with github actions (if repo is on github)

https://learn.microsoft.com/en-us/azure/azure-resource-manager/templates/deploy-github-actions?tabs=userlevel

- use the github action `deploy_dev_resources_gh.yml`
- get subscription and rg id: `az group list`
- create a contributor service principal:

```bash
az ad sp create-for-rbac \
--name "<myrgcontributorsp>" --role contributor \
--scopes /subscriptions/<subscription-id>/resourceGroups/<group-name> \
--sdk-auth
```

- add full json object to github repository secrets under name `AZURE_CREDENTIALS`
- also create other secrets: `AZURE_SUBSCRIPTION` and `AZURE_RG` and AZURE_LOCATION
- add the repository variable `BASE_NAME`

### Deployment with azure devops (if repo is on azure devops)

- no need to create a sp manually, instead go to project settings > service connections
  then choose Azure Resource manager and configure the SP here

- Create a new variable group named mlopsvars (Pipelines / Library) and you can easily add variables.
- Add `AZURE_RG`, `AZURE_LOCATION`, `BASE_NAME` and `AZURE_RM_SVC_CONNECTION` (the name of the connector that was just created)
- Optionally click on the lock to make the variables secrets
- create a pipeline using `deploy_dev_resources_azdvops.yml`

### QoL

- configure defaults: `az configure --defaults group=<myrg> workspace=<myworkspace> location=<location>`

## Track files with dvc

- Run `dvc init`
- Run the get_data.ipynb notebook to download the train and test file and put them
  in data/01_raw

### Local tracking

- in `.dvc/config`, track everything locally:

```
['remote "localdvc"']
    url = ../localdvc
[core]
    autostage = true
    remote = localdvc
```

- Note: `autostage = true` will allow you to automatically stage a file when it is added
  with dvc
- Note2: the config file is saved to source control as it does not contain any sensitive info

### Remote tracking

- if you have a storage on azure instead, create a config.local file by running

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

Note that you can get the connection string: `az storage account show-connection-string --name <storageaccountname> --key key1`

config.local is not committed to source control and is used in priority over config

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

## Create your project with kedro

https://docs.kedro.org/en/stable/get_started/install.html

- to create a new kedro project, run `kedro new`. The created folder is the working
  directory for this repo. Note that i moved the requirments.txt outside of the src folder
- Create pipeline with `kedro create pipeline <my_pipeline>`
- Run a pipeline locally with `kedro run -p <my_pipeline>`

## Add mlflow to kedro

https://kedro-mlflow.readthedocs.io/en/stable/

- To use the kedro-mlflow plugin, run `kedro mlflow init`. This will create a
  mlflow.yml file.
- By default experiments are tracked locally.

- If you are using azure ml, you can change it to point to the aml workspace: you can obtain it
  either by running `mlflow.get_tracking_uri()` from a notebook directly on the workspace or
  by running `az ml workspace show -n <workspacename>`.

## Combine kedro and azure pipelines to run the code on an AML compute cluster

https://kedro-azureml.readthedocs.io/en/0.4.1/source/03_quickstart.html#

- While you can run code directly in the azure ml workspace, you can also develop locally and use the kedro-azureml plugin to run the kedro pipeline on an aml compute cluster. This has the advantage of creating an azure ml pipeline that you can then investigate directly.

### Create a kedro-image to use as a training env for the aml workspace

- Run `kedro docker init`: this will automatically create a Dockerfile, a .dockerignore
  and a .dive-ci file to help with kedro deployment.

- This plugin can helop you run kedro in a docker environment locally but here, we are not
  interested in this functionnality. We just need the base for the azureml pipelines. Therefore,
  in the docker file, remove everything except the requirements part
- Run `kedro docker build --image=<acrname>.azurecr.io/<your_image_name>:latest`

- login to the azure container registry: `az acr login --name <acrname>`
- push the image: `docker push <acrname>.azurecr.io/<your_image_name>:latest`

- Finally, create the azure ml environment:
  `az ml environment create --name <environment-name> --image <acrname>.azurecr.io/<your_image_name>:latest`
  You will be able to see the environment on azure ml

### Use kedro-azureml to launch the training on the workspace

- run `kedro azureml init <AZURE_SUBSCRIPTION_ID> <AZURE_RESOURCE_GROUP> <AML_WORKSPACE_NAME> <EXPERIMENT_NAME> <COMPUTE_NAME> -aml-env <amlenv:version>  -a <STORAGE_ACCOUNT_NAME> -c <STORAGE_CONTAINER_NAME>`

Note that the COMPUTE_NAME must be a cluster instance not a compute instance. Be careful because this works by sending all the data to storage before running the pipeline so we must be careful with the cost management

- This creates
