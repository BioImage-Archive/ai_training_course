# Settings up machine learning environments

    N.b All commands in this box are to be entered into a terminal

## GPU support:

Due to the fast/parallel matrix multiplicative properties of GPUs they have seen a strong co-option into machine learning.
Without a GPU availiable most deep learning and nearly all machine learning on images is a fools errand.
The current industry preference is towards Nvidia's GPUs for DL/ML and as such support for AMD and integrate Intel graphics is limited at best.

### Driver installation:

🐧 Ubuntu:

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt install nvidia-driver-460

🪟 Windows:

    https://www.nvidia.co.uk/Download/index.aspx?lang=en-uk


🍎 MacOs:

Installing Nvidia graphics drivers on MacOS is near impossible (and at best difficult with an eGPU) with most machine learning libraries offering no support for GPU acceleration 

## CUDA

Is Nvidia's proprietary parallel processing software that interfaces with their graphics cards and is crucial for a functioning GPU accelerated ML environment. 

🪟 Windows:
    https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

🐧 Ubuntu:

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pubsudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
    sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'                           
    sudo apt update
    sudo apt install cuda-10–2
    sudo apt install libcudnn7

And check that it didn't fail using:

    nvidia-smi


## Languages

### Python

For all of our examples we will be using python as it is well established for machine- and deep- learning. A version of Python is installed on most modern OSs, though it worth noting these versions can be significantly out of date.

https://www.python.org/download/

🐧 Ubuntu: 

    sudo apt-get install python3.8

🍎 MacOs:

    brew install python@3.8

The very latest versions of Python are not advisable as they can break code-bases and devs need time to ensure compatibility.

### R

R is additionally a feasible language to use however the breadth of the availiable documentation and software for Python makes it incomparable

## Package management

Native Python cannot do nearly as much as the modern ecosystem allows.
As such there is an extensive set of libraries availiable with each being further interdependent on other packages.

### Pip

Pip Installs Packages (pip) for Python and attempts to manage packages' dependencies for you by reading the version requirements of each package recursively.
Pip comes natively with Python installations

Installing a package with Pip:

    pip install numpy

Installing many packages with Pip:

    pip install numpy scipy

### Anaconda

Anaconda is a more holistic and general system wide package manager geared towards science and data science. To this end it can manage Python as well R installations package management.
The stock version of Anaconda all comes preinstalled with a swathe of useful packages.

We recommend using anaconda as (even though it can be slow and unwieldy) it does provide features that are beneficial for machine learning, including driver installations, reproducible environments using desired-state-files and easy-to-use package environment isolation (so that incompatible packages can coexist on the same machine).

🐧 OSs:

    curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh | sh

🍎 MacOS:

    curl https://repo.anaconda.com/archive/Anaconda3-2021.05-MacOSX-x86_64.sh | sh


### Miniconda

Miniconda is the same distribution of anaconda just without any preinstalled packages, this is left to the user.

🍎 MacOS:

    https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

🐧 OSs:

    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


### Conda forge:

It is recommended to also enable conda-forge when using *conda as this gives you access to a much wider selection of packages and tools.

    conda config --add channels conda-forge
    conda config --set channel_priority strict

# Using *conda:

Once correctly installed your terminal will feature a new prefix:

    base ❯

Denoting that we are now in the 'base' conda environment. Be extremely cautious when install packages here as lasting damage to your conda environment can be done and a fresh reinstall of conda will be needed. It is however recommended to install "sensible" packages and can be seen by the entire system here:

We recommend:

    conda install jupyterlab nb_conda_kernels pip

nb_conda_kernels is useful for helping jupyter detecting availiable environments

Let's create a new envionment for machine learning:

    conda create --name ml python=3.8 ipython --yes

We install ipython here so that nb_conda_kernels can detect any new environments we create

And then switch to it:
    conda activate ml

From here we can do one of two things, we can either use conda to install all of the ML packages we want or we can use pip or even a mixture. Conda is extremely diligent (to a fault) when it looks at package interdepenence and in extreme cases can take 10s of minutes to decide what it wants to do. For now we would recommend installing using conda and falling back to pip if needed:


Linux/Windows only:

    conda config --add channels pytorch
    conda install tensorflow pytorch torchvision torchaudio cudatoolkit=10.2

To check this didn't obviously fail:

    python -c "import tensorflow; print(tensorflow.__version__)"
    python -c "import torch; print(torch.__version__)"

## pyTorch vs Tensorflow

pyTorch (Facebook) and Tensorflow (Google) are both machine learning libraries which implement the low level functions needed to build and deploy deep learning models.
pyTorch is younger and with a somewhat cleaner syntax, however a large portion of established models are still Tensorflow based only with no plans to move to pyTorch.


### Check GPU support:

    python -c "import torch;print(torch.cuda.get_device_name(0))"
    python -c "import tensorflow as tf; tf.test.gpu_device_name()"

## Useful additional packages

|   |   |
|---|---|
|https://scikit-learn.org/ |A high quality machine learning library that is geared towards learning frame tables and DataFrames|
|https://keras.io/|Attempts to standard the machine learning and model building process across multiple different backends, widely used.|
|https://www.fast.ai/|Another higher level library (like Keras), uses pyTorch
|
|https://skorch.readthedocs.io/|Marries pyTorch models to the sckit-learn interface|


## Jupyter

Ju[Lia]Py[thon]e[R] provides a web-based interface to an interactive coding environment originally aimed at the Julia Python and R languages but now including more including Java, Sage, Matlab (https://github.com/jupyter/jupyter/wiki/Jupyter-kernels) and more.
The advantage of interaction coding environments is the ability ot explore and develope dynamically.

To run a jupyter notebook:
    jupyter notebook


