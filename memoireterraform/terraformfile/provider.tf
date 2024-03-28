terraform {
    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 5.0"
        }
    }
  
}
variable "AWS_ACCESS_KEY" {}
variable "AWS_SECRET_KEY" {}

provider "aws"{
    region = "us-east-2"
    access_key = var.AWS_ACCESS_KEY
    secret_key = var.AWS_SECRET_KEY
}