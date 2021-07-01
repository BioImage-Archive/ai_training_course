# Base image to work from that is pulled from Dockerhub,
# [Image:tag] format
FROM python:3.7
# Download git repo into the image
RUN git clone -b main https://github.com/google/microscopeimagequality.git
# Switch working directory
WORKDIR "./microscopeimagequality"
# Install dependencies
RUN pip3 install --editable .
# Download the pretrained model
RUN microscopeimagequality download 
#Add microscopeimagequality to the path
ENV PYTHONPATH="${PYTHONPATH}:~/microscopeimagequality"
#Make the microscopeimagequality application the default command on running
ENTRYPOINT ["microscopeimagequality"]