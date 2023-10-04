param location string
param aml_workspace string
param name string

resource mlwgpu 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  name: '${aml_workspace}/${name}'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      osType: 'Linux'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 2
        nodeIdleTimeBeforeScaleDown: 'PT5M' // RFC format
      }
      subnet: null
      vmSize: 'Standard_NC4as_T4_v3'
    }
  }
}

output mlwgpuOut string = mlwgpu.id
