data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

data "aws_key_pair" "existing_key_pair" {
  key_name = "namberpython"  # Replace with the name of your existing key pair
}

resource "aws_instance" "website" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"
  vpc_security_group_ids = [ aws_security_group.my_instance_SG.id ]
  key_name = data.aws_key_pair.existing_key_pair.key_name  # Use the existing key pair name
  
  user_data = <<-EOF
              #!/bin/bash
              


              sudo apt-get update
              sudo apt install docker.io -y
              sudo apt install python3-pip -y
              sudo pip3 install psutil flask
              

              sudo systemctl restart docker
              sudo usermod -aG docker 
              sudo apt install git -y
              sudo usermod -aG git
              sudo git clone https://github.com/tallahmad047/projetflask.git
             

                       


              cd projetflask/
              sudo docker build -t myflaskappv1 .
              sudo docker run -p 5000:5000 myflaskappv1
              
              EOF

            

  tags = {
    Name = "my-python-app"
  }
}