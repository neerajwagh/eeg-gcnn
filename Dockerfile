FROM nvidia/cuda:11.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


# Install some basic utilities
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    gcc \
    git \
    bzip2 \
    libx11-6 \
    apt-utils \
    unzip \
    tar \
    python3-opencv \
    libsvm-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
&& chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

RUN conda create -n mne_pyt python=3.8

RUN echo "source activate mne_pyt" >> ~/.bashrc

RUN /bin/bash -c "source ~/.bashrc"

RUN pip install scipy

RUN pip install numpy

RUN pip install scikit-learn

RUN pip install pandas

RUN pip install jupyter

RUN pip install h5py

RUN pip install graphviz

RUN pip install pydot

RUN pip install keras

RUN pip install matplotlib

RUN pip install seaborn

RUN pip install joblib

RUN pip install -U mne

RUN conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch

RUN pip install torch-scatter==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-sparse==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-cluster==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-spline-conv==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-geometric==1.7.2
RUN pip install captum
RUN pip install tensorboard

RUN echo 'alias python="/home/user/miniconda/bin/python"' >> /home/user/.bashrc

# set environment variables
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib_cache

# What should we run when the container is launched
ENTRYPOINT ["/bin/bash"]





