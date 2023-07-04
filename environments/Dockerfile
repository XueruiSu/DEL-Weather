ARG BASE_IMAGE=openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:latest
FROM mcr.microsoft.com/azureml/${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    gpg \
    curl \
    vim \
    wget \
    ca-certificates \
    sysstat \
    net-tools \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    openssh-client \
    openssh-server \
    libsm6 \
    libxext6 \
    ffmpeg \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set bash, as conda doesn't like dash:
SHELL [ "/bin/bash", "--login", "-c" ]

# Make bash aware of conda:
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.profile

# Default to python 3.8 (and don't only do this as part of the image, because it breaks
# conda's dependency resolver):
RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge "python==3.9.13" mamba

# Install the extra dependencies into the default image, so that downstream
# users do not need to choose the container AND switch images:
COPY ./climai_global-environment.yml /tmp/
ENV PIP_NO_CACHE_DIR=1

RUN mamba env update -p /opt/miniconda -f /tmp/climai_global-environment.yml && conda clean --all --yes