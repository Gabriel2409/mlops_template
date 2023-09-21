param name string
param location string
param tags object

resource cr 'Microsoft.ContainerRegistry/registries@2023-06-01-preview' = {
  name: name
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
