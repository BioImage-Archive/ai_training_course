BASEDIR = $(shell pwd)
# https://github.com/NVIDIA/nvidia-container-runtime

all: test_docker test_cuda_demo

test_docker: test_docker_gpu

test_cuda_demo: build_cuda_demo test_cuda_gpu_demo

test_cuda_gpu_demo: test_docker_tensorflow_in_cuda_demo test_docker_pytorch_in_cuda_demo test_docker_gpu_in_cuda_demo

build_cuda_demo:    
	docker build -t cuda_demo:latest .

test_docker_gpu:
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl nvidia/cuda:11.0-base nvidia-smi

test_docker_gpu_in_cuda_demo:    
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl cuda_demo nvidia-smi


test_docker_pytorch_in_cuda_demo:    
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl \
		cuda_demo conda run --name torch python -c "import torch; print(torch.__version__)"

test_docker_pytorch_in_cuda_demo_gpu:    
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl \
		cuda_demo conda run --name torch python -c "import torch;print(torch.cuda.get_device_name(0))"


test_docker_tensorflow_in_cuda_demo:    
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl \
		cuda_demo conda run --name tf python -c "import tensorflow; print(tensorflow.__version__)"


test_docker_tensorflow_in_cuda_demo_gpu:    
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl \
		cuda_demo conda run --name tf python -c "import tensorflow as tf; tf.test.gpu_device_name()"

ssh_in_cuda_demo:    
	docker run --gpus all --device /dev/nvidia0 --device \
		/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device \
		/dev/nvidiactl -it --rm --gpus all --entrypoint /bin/bash cuda_demo 

play_cast:
	asciinema play install_env.cast