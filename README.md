TF-Kube: Infrastructure as Code with Terraform and Kubernetes
TF-Kube is a project that showcases the integration of Terraform for infrastructure provisioning and Kubernetes for deploying containerized applications, all managed through an automated CI/CD pipeline in Azure DevOps. The project includes a user-facing health query interface, likely developed in collaboration with the Mount Sinai Health System, allowing users to ask health-related questions such as "What is Tylenol?" This repository provides the tools and configuration necessary to set up, deploy, and manage the infrastructure and application components.

Features
Infrastructure as Code (IaC): Provision and manage cloud resources using Terraform.
Container Orchestration: Deploy and scale applications on Kubernetes.
CI/CD Automation: Streamline build, deployment, and destruction processes with Azure DevOps pipelines.
Health Query Interface: Enable users to query health-related information through a dedicated frontend, integrated with backend services.
Scalable Deployment: Utilize NGINX ingress and Kubernetes services for robust application access.
Technologies Used
Azure DevOps: For CI/CD pipelines, including Azure Repos, Build Pipeline, and Release Pipeline.
Terraform: For defining and managing infrastructure as code.
Kubernetes: For container orchestration and workload management.
Azure Subscription: For hosting the backend and cloud resources.
NGINX Ingress: For external traffic routing to Kubernetes services.
Visual Studio Code: Recommended IDE for development.
Project Structure
The project leverages a modular architecture, with key components including:

CI/CD Pipeline: Automates the build, deployment, and destruction of infrastructure and applications.
Kubernetes Workloads: Deploys backend-deployment, frontend-deployment, and ollama-deployment for application functionality.
Health Query Interface: A user interface connected to the Mount Sinai Health System, providing health-related responses.
CI/CD Pipeline Overview
The pipeline is configured in Azure DevOps and consists of the following stages:

Build Job
Get Source: Retrieves code from Azure Repos.
Install Terraform: Installs the latest Terraform version.
Terraform Init: Initializes the Terraform working directory.
Terraform Validate: Validates configuration files.
Terraform Plan: Generates an execution plan.
Archive Files: Packages artifacts for deployment.
Publish Artifacts: Publishes artifacts for the release pipeline.
Release Pipeline
Trigger Release: Initiates the release process upon successful build.
Get Artifacts: Downloads build artifacts.
Deploy Stage:
Download and extract artifacts.
Install Terraform.
Initialize Terraform.
Apply Terraform configuration to deploy resources.
Manual approval step for deployment validation.
Destroy Stage:
Download and extract artifacts.
Install Terraform.
Initialize Terraform.
Destroy deployed resources.
Kubernetes Workloads and Services
The project deploys the following workloads and services on Kubernetes:

Workloads:
backend-deployment: Handles backend logic (default namespace).
frontend-deployment: Serves the health query interface (default namespace).
ollama-deployment: Possibly an AI or supplementary service (default namespace).
System workloads: coredns, metrics-server, etc. (kube-system namespace).
Services:
backend, frontend, ollama: ClusterIP services for internal communication.
ingress-nginx-controller: LoadBalancer service with an external IP (e.g., 130.107.165.59) for public access.
Health Query Interface
The health query interface, branded with the Mount Sinai Health System, allows users to input questions (e.g., "What is Tylenol?") and receive detailed responses about medications, including usage, precautions, and contraindications. Example output includes:

Tylenol Description: An OTC pain reliever and fever reducer containing acetaminophen.
Forms: Tablets, capsules, liquids, gels.
Precautions: Not suitable for children under 6, pregnant/breastfeeding women, or individuals with liver disease.
Setup and Installation
Prerequisites
Azure CLI: For interacting with Azure services.
Terraform: For infrastructure management.
kubectl: For Kubernetes cluster management.
Azure DevOps Account: For pipeline configuration and execution.
Azure Subscription: For hosting resources.
Installation Steps
Clone the Repository:
bash

Copy
git clone https://github.com/Sayan8967/myaiwebapp.git
cd TF-Kube
Install Dependencies:
Install Azure CLI: Official Guide.
Install Terraform: Download.
Install kubectl: Official Guide.
Configure Environment:
Log in to Azure:
bash

Copy
az login
Set up Terraform backend (e.g., Azure Blob Storage) in your configuration files.
Usage and Deployment
Initialize Terraform
bash

Copy
terraform init
Plan Infrastructure
bash

Copy
terraform plan
Deploy Infrastructure
bash

Copy
terraform apply
Manage Kubernetes Resources
Check cluster status:
bash

Copy
kubectl get pods -A
Access the health query interface via the NGINX ingress external IP.
Run the CI/CD Pipeline
Push changes to Azure Repos.
Trigger the build pipeline in Azure DevOps.
Monitor the release pipeline for deployment or destruction.
Destroy Resources
bash

Copy
terraform destroy
Contributing
We welcome contributions to enhance TF-Kube! To contribute:

Fork the repository.
Create a feature branch:
bash

Copy
git checkout -b feature-branch
Commit your changes:
bash

Copy
git commit -m "Add new feature"
Push to your branch:
bash

Copy
git push origin feature-branch
Open a pull request for review.
License
This project is licensed under the MIT License. See the  file for details.

Acknowledgments
Mount Sinai Health System for collaboration on the health query interface.
Azure DevOps team for robust CI/CD tools.
Terraform and Kubernetes communities for their open-source contributions.

Notes
Customize the external IP, specific Terraform backend configuration, or additional pipeline details as needed based on your project's specifics.
If the Google Slides link contains additional details (e.g., specific versions, unique features), integrate them upon review.

Google slides: https://docs.google.com/presentation/d/1QP4-VIkojWmcNwSCMKHn1waGA378NOWY9jralBVnhX8/edit?usp=sharing
