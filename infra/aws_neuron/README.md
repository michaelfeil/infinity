# Launch an EC2 Instance on AWS:

### Start a EC2 Instance with Huggingface AMI (free AMI image with Neuron Tools/Docker installed)
- https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2
- View Purchase Options -> Configure
- Use `64-Bit AMI`, `20241115 (Nov 18, 2024)`
- Region, e.g. `us-west-2`
- Set Instance type `inf2.xlarge` (has two neuron accelerators)
- Login with username `ubuntu` (using your standard EC2 setup e.g. `ssh ubuntu@ec2-14-11-13-12.us-west-2.compute.amazonaws.com`)

### Run

```bash
docker run -it -rm --name infbase --entrypoint /bin/bash --device=/dev/neuron0 --pull always michaelf34/aws-neuron-base-img:inf-new
```