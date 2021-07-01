FROM python:3.7
RUN apt-get update
RUN git clone -b main https://github.com/google/microscopeimagequality.git
WORKDIR "./microscopeimagequality"
RUN python --version
RUN pip3 install --editable .
RUN microscopeimagequality download 
ENV PYTHONPATH="${PYTHONPATH}:~/microscopeimagequality"
ENTRYPOINT ["microscopeimagequality"]