FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && \
      apt-get -y install sudo curl wget

RUN sudo apt-get install python3.8 python3-pip --yes

## Setup Conda

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

RUN bash ~/miniconda.sh -b -p /opt/conda && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&     echo "conda activate base" >> ~/.bashrc &&     find /opt/conda/ -follow -type f -name '*.a' -delete &&     find /opt/conda/ -follow -type f -name '*.js.map' -delete &&     /opt/conda/bin/conda clean -afy

## 

RUN conda config --add channels conda-forge && conda config --set channel_priority strict

RUN conda install jupyterlab nb_conda_kernels pip --yes

# RUN conda create --name ml python=3.8 ipython ipykernel --yes

RUN conda config --add channels pytorch

# RUN conda config --add channels anaconda

RUN conda create python ipython ipykernel tensorflow-gpu pytorch torchvision torchaudio cudatoolkit --name ml --yes

RUN conda create python ipython ipykernel tensorflow-gpu -c anaconda --name tf --yes

RUN conda create python ipython ipykernel pytorch torchvision torchaudio cudatoolkit --name torch --yes

# RUN conda run --name torch python -c "import torch; print(torch.__version__)"

# RUN conda run --name torch python -c "import torch;print(torch.cuda.get_device_name(0))"

# RUN conda run --name tf python -c "import tensorflow; print(tensorflow.__version__)"

# RUN conda run --name tf python -c "import tensorflow as tf; tf.test.gpu_device_name()"
