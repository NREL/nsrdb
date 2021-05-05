# syntax=docker/dockerfile:1.0.0-experimental

## For this to work you must run `export DOCKER_BUILDKIT=1`
## then build using the command
##  docker build --ssh github_ssh_key=/Users/<your_username>/.ssh/id_rsa .

FROM public.ecr.aws/lambda/python:3.8
# Set working directory
WORKDIR /nsrdb

# download public key for github.com
RUN yum install -y openssh-server openssh-clients

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install Rest2
RUN --mount=type=ssh,id=github_ssh_key pip install git+ssh://github.com/NREL/rest2

# Copy package
COPY . /nsrdb
# Install package and dependencies
RUN pip install .

CMD ["lambda.handler"]
