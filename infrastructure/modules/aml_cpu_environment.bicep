param aml_workspace string
param name string
param condaFile string

resource cpuenv 'Microsoft.MachineLearningServices/workspaces/environments@2023-06-01-preview' = {
  name: '${aml_workspace}/${name}'
  properties: {
    description: 'Cpu environment - based on Microsoft cpu images + conda dependencies'
    properties: {
      image: 'mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest'
      condaFile: condaFile
    }
  }
}
