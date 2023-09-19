param location string
param name string

resource mlwcc 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  name: name
  location: location
  properties: {
    computeType: 'AmlCompute'
    osType: 'Linux'
    scaleSettings: {
      minNodeCount: 0
      maxNodeCount: 2
      nodeIdleTimeBeforeScaleDown: 1800
    }
    subnet: null
    vmSize: 'Standard_DS3_v2'
  }
}
