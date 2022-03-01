FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

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

RUN conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-geometric
RUN pip install captum
RUN pip install tensorboard



