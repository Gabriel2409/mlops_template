param location string
param aml_workspace string
param name string

resource mlwcc 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  name: '${aml_workspace}/${name}'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      osType: 'Linux'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 2
        nodeIdleTimeBeforeScaleDown: '1800s'
      }
      subnet: null
      vmSize: 'Standard_DS3_v2'
    }
  }
}

output mlwccOut string = mlwcc.id
