terraform {
  backend "azurerm" {
    resource_group_name  = "SA-Res"
    storage_account_name = "storage896"
    container_name       = "prod-tfstate"
    key                  = "prod.terraform.tfstate"
  }
}
