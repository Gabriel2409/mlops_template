
AZURE_RG="$1"
AZUREML_WORKSPACE="$2"
TRAINING_KEDRO_PIPELINE="$3"

echo "There is probably a better way to retrieve the job id corresponding to the azureml pipeline"
parent_pipeline=$(az ml job  list -w $AZUREML_WORKSPACE -g $AZURE_RG --query "[?type=='pipeline']|[?creation_context.created_by_type=='Application']|[?display_name=='${TRAINING_KEDRO_PIPELINE}']|sort_by(@, &creation_context.created_at)[-1].name" -o tsv)
echo "Retrieving pipeline: $parent_pipeline"
job_id=$(az ml job list -w $AZUREML_WORKSPACE -g $AZURE_RG -p $parent_pipeline --query "[?contains(not_null(tags.node_names,''),'log_sklearn_scores')]|[0].name" -o tsv)
echo "Retrieving job: $job_id"
az ml model create --name $TRAINING_KEDRO_PIPELINE --path runs:/$job_id/model/ --type mlflow_model -g $AZURE_RG -w $AZUREML_WORKSPACE