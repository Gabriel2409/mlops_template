param baseName string
param location string
param tags object

resource cr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'cr-${baseName}'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
  tags: tags
}

output crOut string = cr.id
