resource "aws_security_group" "my_instance_SG" {
  name_prefix = "dele-"

tags = {
    Name="my-security-group"
}


}

# Allow incoming HTTP (port 80) traffic
resource "aws_security_group_rule" "http_inbound" {
  type        = "ingress"
  from_port   = 80
  to_port     = 80
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]

  security_group_id = aws_security_group.my_instance_SG.id
}

# Allow incoming SSH (port 22) traffic
resource "aws_security_group_rule" "ssh_inbound" {
  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]

  security_group_id = aws_security_group.my_instance_SG.id
}
# Allow incoming Flsk/python (port 5000) traffic
resource "aws_security_group_rule" "flask_inbound" {
  type        = "ingress"
  from_port   = 5000
  to_port     = 5000
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]

  security_group_id = aws_security_group.my_instance_SG.id
}
resource "aws_security_group_rule" "downloads" {
  type        = "egress"
  from_port   = 0
  to_port     = 0
  protocol    = "-1"
  cidr_blocks = ["0.0.0.0/0"]

  security_group_id = aws_security_group.my_instance_SG.id
}