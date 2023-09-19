# Mlops template

## Resources
- https://madewithml.com/ + wayback machine (july 23 for old course). Data from here
- https://github.com/Azure/mlops-v2: used to create a fast template for azureml + azuredevops

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
  `az deployment group create --resource-group <myrg> --template-file infrastructure/main.bicep --parameters infrastructure/dev_resources/default_parameters.bicepparam`

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
- also create other secrets: `AZURE_SUBSCRIPTION` and `AZURE_RG` and `AZURE_LOCATION`
- add the repository variable `BASE_NAME`

### Deployment with azure devops (if repo is on azure devops)

- no need to create a SP manually, instead go to project settings > service connections
  then choose Azure Resource manager and configure the SP here
- Note: if you go to `App registrations` in azure portal, you should see the service principal

- Create a new variable group named mlopsvars (Pipelines / Library) and you can easily add variables.
- Add `AZURE_RG`, `AZURE_LOCATION`, `BASE_NAME` and `AZURE_RM_SVC_CONNECTION` (the name of the connector that was just created)
- Optionally click on the lock to make the variables secrets
- create a pipeline using `deploy_dev_resources_azdvops.yml`

### QoL

- configure defaults: `az configure --defaults group=<myrg> workspace=<myworkspace> location=<location>`

## Track files with dvc (skip if using azureml data asset)

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

- If you are using azure ml, you can change it to point to the aml workspace: you can obtain it either by running `mlflow.get_tracking_uri()` from a notebook directly on the workspace or by running `az ml workspace show -n <workspacename>`. Note that when using the ui with `kedro mlflow ui`, you MUST also set the `MLFLOW_TRACKING_URI` env var to point to the same value. For some reason, artifacts are not loaded correctly if this var is not set

- Note that if you launch your script on an azure ml compute instance or compute cluster, you don't need to modify anything (which is why the mlflow.yml file is not committed to source control)

## Create a custom environment to use for your compute instances and clusters

- By default, azure comes with a lot of pre-configured environments that can be used as is.
- It is also possible to create an environment from another by adding conda dependencies.
- Finally, you can create your own environment from a dockerfile and push it to your
  container registry.
- very nice resource: https://bea.stollnitz.com/blog/aml-environment/

### kedro-docker (skip if using environment)

- plugin to help create kedro docker images
- we are only interested in creating the environment so the process is even easier here
- Run `kedro docker init`: this will automatically create a Dockerfile, a .dockerignore
  and a .dive-ci file to help with kedro deployment.
- In the docker file, remove everything except the requirements part
- Run `kedro docker build --image=<acrname>.azurecr.io/<your_image_name>:latest`

- login to the azure container registry: `az acr login --name <acrname>`
- push the image: `docker push <acrname>.azurecr.io/<your_image_name>:latest`

- Finally, create the azure ml environment:
  `az ml environment create --name <environment-name> --image <acrname>.azurecr.io/<your_image_name>:latest`
  You will be able to see the environment on azure ml

### From an existing image / environment

- It is also possible to go to the azure ml studio, start from a list of curated
  environment and customize them.
- It is also possible to start from a list of pre-built microsoft images:
  https://github.com/Azure/AzureML-Containers. In this case, you can easily specify
  a conda-dependencies file and register a new environment

```yml
# ex cond-dependencies.yaml
name: conda_env_test
dependencies:
  - python==3.10.11
  - pip=23.2.1
  - pip:
      - pandas==1.1.3
      - scikit-learn~=1.3.0
```

```python
# register a new environment with python sdk
from azureml.core.environment import Environment
from azureml.core import Workspace
ws = Workspace(
    subscription_id="<subscription-id>",
    resource_group="<rg>",
    workspace_name="<ws>",
)

env = Environment.from_docker_image(
    name='conda_env_test',
    image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest',
    conda_specification= "conda-dependencies.yaml"
)
env.register(workspace=ws)
```

- or from the cli:

`az ml environment create --resource-group <rg> --workspace-name <ws> --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest --conda-file conda-dependencies.yaml --name <env_name>`


- To automate this, you can also create a pipeline from `infrastructure/aml_environments/create_cpu_env.yaml`


- TODO: Microsoft also has a list of prebuilt images for inference. It might be worth checking out

## Create a compute cluster for jobs and ml pipelines

- We can create a compute cluster directly from the azure ml studio

- Alternatively, we can use the python sdk

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "mycluster"
try:
    cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2", max_nodes=4, idle_seconds_before_scaledown=2400
    )
    cluster = ComputeTarget.create(ws, cluster_name, compute_config)
cluster.wait_for_completion(show_output=True)
```

- We can also use the cli

```yml
# cluster_specs.yml
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json
name: mycluster
type: amlcompute
size: STANDARD_DS3_v2
min_instances: 0
max_instances: 2
idle_time_before_scale_down: 1800
tier: dedicated
```

Then run `az ml compute create --resource-group <rg> --workspace-name <ws> --file $script_dir/cluster_specs.yml`


- To automate this, you can also create a pipeline from `infrastructure/aml_compute_clusters/create_dev_compute_cluster.yml`


## Submit an azure ml job

- You can submit jobs and pipelines to azure ml

- python sdk

```python
from azureml.core.script_run_config import ScriptRunConfig
from azureml.core import Experiment

exp = Experiment(ws, 'myexp')

config = ScriptRunConfig(
    source_directory='./path/to/dir/',
    script='myscript.py',
    compute_target=cluster,
    environment=env,
)

run = exp.submit(config)
run.wait_for_completion(show_output=True)
```

- cli version, example with kedro

```yml
#example_job.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
description: 'Run Kedro'
# experiment_name: "myexp" # experiment is already set with kedro, including an experiment_name will lead to an error
compute: azureml:cpu-dev-cluster # name of the compute cluster
command: |
  kedro run
environment: azureml:mlopstpl-cpu-env@latest #name of the custom environment
code: . # directory to include (files in .gitignore and .amlignore are not included)
```

Run your job with: `az ml job create --file example_job.yaml -w <workspace> -g <rg>`

Note: Code used is stored on a container in the blob storage associated to a workspace.

## Combine kedro and azure pipelines to run the code as a pipeline instead of a job

- The problem with the current setup is that the full kedro pipeline is run as a single
  job on a given cluster. The kedro-azureml plugin allows to run the code as an azure ml
  pipeline where each node in the pipeline corresponds to a different step. Moreover,
  each node can be run on a different cluster. More info here:
  https://kedro-azureml.readthedocs.io/en/0.4.1/source/03_quickstart.html#

- Below is an example on how to use the plugin

- run `kedro azureml init <AZURE_SUBSCRIPTION_ID> <AZURE_RESOURCE_GROUP> <AML_WORKSPACE_NAME> <EXPERIMENT_NAME> <COMPUTE_NAME> --aml-env <amlenv:version>  -a <STORAGE_ACCOUNT_NAME> -c <STORAGE_CONTAINER_NAME>`

Note that the COMPUTE_NAME must be a cluster instance not a compute instance.
Be careful because this works by sending all the data to storage before running the pipeline so we must be careful with the cost management. `Customize the .amlignore` file to exclude everything that you don't need for compute. Note that with kedro-azureml, `.amlignore` is not combined with your `.gitignore`

