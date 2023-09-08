
AZURE_RG="$1"
AZUREML_WORKSPACE="$2"
TRAINING_KEDRO_PIPELINE="$3"


parent_pipeline=$(az ml job  list -w $AZUREML_WORKSPACE -g $AZURE_RG --query "[?type=='pipeline']|[?creation_context.created_by_type=='Application']|[?display_name=='${TRAINING_KEDRO_PIPELINE}']|sort_by(@, &creation_context.created_at)[-1].name" -o tsv)
job_id=$(az ml job list -w $AZUREML_WORKSPACE -g $AZURE_RG -p parent_pipeline --query "[?contains(not_null(tags.node_names,''),'log_sklearn_scores')]|[0].name" -o tsv)
az ml model create --name $TRAINING_KEDRO_PIPELINE --path runs:/$job_id/model/ --type mlflow_model -g $AZURE_RG -w $AZUREML_WORKSPACE