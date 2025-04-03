
# Create a resource group
resource "azurerm_resource_group" "example" {
  name     = "webapp-resource"
  location = "East US"
}

# Provision an AKS cluster
resource "azurerm_kubernetes_cluster" "example" {
  name                = "webapp-aks"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name
  dns_prefix          = "exampleaks"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D4ds_v5" # 2 CPUs, 7GB RAM, sufficient for the Ollama resource requests
  }

  identity {
    type = "SystemAssigned"
  }
}

# Configure the Kubernetes provider to connect to the AKS cluster
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.example.kube_config.0.host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.example.kube_config.0.client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.example.kube_config.0.client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.example.kube_config.0.cluster_ca_certificate)
}

# Write the AKS kubeconfig to a local file
resource "local_file" "kubeconfig" {
  content  = azurerm_kubernetes_cluster.example.kube_config_raw
  filename = "${path.module}/kubeconfig"
}

# Apply the NGINX Ingress Controller YAML using kubectl
resource "null_resource" "apply_ingress_controller" {
  provisioner "local-exec" {
    command = "kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml --kubeconfig=${local_file.kubeconfig.filename}"
  }
  depends_on = [
    azurerm_kubernetes_cluster.example,
    local_file.kubeconfig
  ]
}

# Wait for NGINX Ingress Controller to be ready
resource "null_resource" "wait_for_ingress_controller" {
  provisioner "local-exec" {
    interpreter = ["powershell", "-Command"]
    command     = <<EOT
      $timeout = 300  # 5 minutes
      $interval = 5
      $elapsed = 0
      while ($elapsed -lt $timeout) {
        $ready = kubectl get pods -n ingress-nginx -l app.kubernetes.io/component=controller --kubeconfig=./kubeconfig -o json | ConvertFrom-Json | Select-Object -ExpandProperty items | ForEach-Object { $_.status.conditions | Where-Object { $_.type -eq "Ready" -and $_.status -eq "True" } }
        if ($ready) {
          Write-Host "NGINX Ingress Controller is ready."
          break
        }
        Write-Host "Waiting for NGINX Ingress Controller to be ready..."
        Start-Sleep -Seconds $interval
        $elapsed += $interval
      }
      if ($elapsed -ge $timeout) {
        Write-Error "Timeout waiting for NGINX Ingress Controller to be ready."
        exit 1
      }
EOT
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Ollama Deployment
resource "kubernetes_deployment" "ollama" {
  metadata {
    name = "ollama-deployment"
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "ollama"
      }
    }
    template {
      metadata {
        labels = {
          app = "ollama"
        }
      }
      spec {
        container {
          name  = "ollama"
          image = "sayan896/project-ollama:latest"
          port {
            container_port = 11434
          }
          volume_mount {
            name       = "ollama-data"
            mount_path = "/root/.ollama"
          }
          readiness_probe {
            http_get {
              path = "/api/tags"
              port = 11434
            }
            initial_delay_seconds = 120
            period_seconds        = 10
          }
          resources {
            requests = {
              memory = "5Gi"
              cpu    = "2"
            }
            limits = {
              memory = "6Gi"
              cpu    = "2"
            }
          }
        }
        volume {
          name = "ollama-data"
          empty_dir {}
        }
      }
    }
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Ollama Service
resource "kubernetes_service" "ollama" {
  metadata {
    name = "ollama"
  }
  spec {
    selector = {
      app = "ollama"
    }
    port {
      protocol    = "TCP"
      port        = 11434
      target_port = 11434
    }
    type = "ClusterIP"
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Backend Deployment
resource "kubernetes_deployment" "backend" {
  metadata {
    name = "backend-deployment"
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "backend"
      }
    }
    template {
      metadata {
        labels = {
          app = "backend"
        }
      }
      spec {
        container {
          name  = "backend"
          image = "sayan896/project-backend:01"
          port {
            container_port = 8000
          }
          env {
            name  = "OLLAMA_URL"
            value = "http://ollama:11434/api/generate"
          }
          env {
            name  = "ALLOWED_ORIGINS"
            value = "*"
          }
        }
      }
    }
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Backend Service
resource "kubernetes_service" "backend" {
  metadata {
    name = "backend"
  }
  spec {
    selector = {
      app = "backend"
    }
    port {
      protocol    = "TCP"
      port        = 8000
      target_port = 8000
    }
    type = "ClusterIP"
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Frontend Deployment
resource "kubernetes_deployment" "frontend" {
  metadata {
    name = "frontend-deployment"
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "frontend"
      }
    }
    template {
      metadata {
        labels = {
          app = "frontend"
        }
      }
      spec {
        container {
          name  = "frontend"
          image = "sayan896/project-frontend:05"
          port {
            container_port = 80
          }
          env {
            name  = "BACKEND_URL"
            value = "/api"
          }
        }
      }
    }
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Frontend Service
resource "kubernetes_service" "frontend" {
  metadata {
    name = "frontend"
  }
  spec {
    selector = {
      app = "frontend"
    }
    port {
      protocol    = "TCP"
      port        = 80
      target_port = 80
    }
    type = "ClusterIP"
  }
  depends_on = [null_resource.apply_ingress_controller]
}

# Ingress for routing traffic
resource "kubernetes_ingress_v1" "app" {
  metadata {
    name = "app-ingress"
    annotations = {
      "nginx.ingress.kubernetes.io/rewrite-target"        = "/$2"
      "nginx.ingress.kubernetes.io/use-regex"             = "true"
      "nginx.ingress.kubernetes.io/proxy-read-timeout"    = "300"
      "nginx.ingress.kubernetes.io/proxy-connect-timeout" = "300"
    }
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "frontend"
              port {
                number = 80
              }
            }
          }
        }
        path {
          path      = "/api(/|$)(.*)"
          path_type = "ImplementationSpecific"
          backend {
            service {
              name = "backend"
              port {
                number = 8000
              }
            }
          }
        }
      }
    }
  }
  depends_on = [null_resource.wait_for_ingress_controller]
}