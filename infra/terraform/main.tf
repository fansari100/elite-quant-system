# Elite Quant System — Terraform Infrastructure
# Multi-cloud deployment on AWS + GCP

terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.30" }
    google = { source = "hashicorp/google", version = "~> 5.10" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.25" }
    helm = { source = "hashicorp/helm", version = "~> 2.12" }
  }
  backend "s3" {
    bucket = "elite-quant-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

# ─── AWS Provider ───────────────────────────────────────────
provider "aws" {
  region = var.aws_region
}

# ─── GCP Provider ───────────────────────────────────────────
provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# ─── Variables ──────────────────────────────────────────────
variable "aws_region" { default = "us-east-1" }
variable "gcp_project" { default = "elite-quant-prod" }
variable "gcp_region" { default = "us-central1" }
variable "environment" { default = "production" }
variable "cluster_name" { default = "elite-quant-cluster" }

# ─── AWS EKS Cluster ───────────────────────────────────────
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_role.arn
  version  = "1.29"

  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.eks.id]
  }

  tags = { Environment = var.environment, ManagedBy = "terraform" }
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.node_role.arn
  instance_types  = ["p4d.24xlarge"]
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = 2
    max_size     = 8
    min_size     = 1
  }

  labels = { "nvidia.com/gpu" = "true", workload = "ml-training" }
}

# ─── AWS VPC ───────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "${var.cluster_name}-vpc" }
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = "${var.aws_region}${["a", "b", "c"][count.index]}"
  tags = { Name = "${var.cluster_name}-private-${count.index}" }
}

resource "aws_security_group" "eks" {
  vpc_id = aws_vpc.main.id
  ingress { from_port = 443; to_port = 443; protocol = "tcp"; cidr_blocks = ["10.0.0.0/16"] }
  egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] }
}

# ─── IAM Roles ─────────────────────────────────────────────
resource "aws_iam_role" "eks_role" {
  name = "${var.cluster_name}-eks-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow",
      Principal = { Service = "eks.amazonaws.com" } }]
  })
}

resource "aws_iam_role" "node_role" {
  name = "${var.cluster_name}-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow",
      Principal = { Service = "ec2.amazonaws.com" } }]
  })
}

# ─── GCP GKE Cluster (multi-cloud failover) ────────────────
resource "google_container_cluster" "failover" {
  name     = "${var.cluster_name}-gcp"
  location = var.gcp_region

  initial_node_count       = 1
  remove_default_node_pool = true

  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.main.name
}

resource "google_container_node_pool" "gpu_pool" {
  name     = "gpu-pool"
  cluster  = google_container_cluster.failover.name
  location = var.gcp_region

  node_config {
    machine_type = "a2-highgpu-1g"
    guest_accelerator { type = "nvidia-tesla-a100"; count = 1 }
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  autoscaling { min_node_count = 0; max_node_count = 4 }
}

resource "google_compute_network" "main" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.1.0.0/20"
  region        = var.gcp_region
  network       = google_compute_network.main.id
}

# ─── Helm Releases (monitoring stack) ──────────────────────
provider "kubernetes" {
  host                   = aws_eks_cluster.main.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.main.certificate_authority[0].data)
}

provider "helm" {
  kubernetes { host = aws_eks_cluster.main.endpoint }
}

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  create_namespace = true

  set { name = "grafana.enabled"; value = "true" }
  set { name = "grafana.adminPassword"; value = "admin" }
}

resource "helm_release" "nginx_ingress" {
  name       = "ingress-nginx"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress"
  create_namespace = true
}

# ─── Outputs ───────────────────────────────────────────────
output "eks_endpoint" { value = aws_eks_cluster.main.endpoint }
output "gke_endpoint" { value = google_container_cluster.failover.endpoint }
