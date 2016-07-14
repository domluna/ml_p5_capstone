# A Dockerfile that sets up a full Gym install
FROM ubuntu:14.04

# WORKDIR /code

RUN apt-get update
RUN apt-get -y install xorg-dev
RUN apt-get -y install libgl1-mesa-dev
RUN apt-get -y install xvfb
RUN apt-get -y install libxinerama1
RUN apt-get -y install libxcursor1
RUN apt-get -y install libglu1-mesa
RUN apt-get -y install libav-tools
RUN apt-get -y install python-numpy
RUN apt-get -y install python-scipy
RUN apt-get -y install python-pyglet
RUN apt-get -y install python-setuptools
RUN apt-get -y install libpq-dev
RUN apt-get -y install libjpeg-dev
RUN apt-get -y install curl
RUN apt-get -y install cmake
RUN apt-get -y install git
#
# For the doom environments
RUN apt-get -y install -y python-dev cmake zlib1g-dev python-opengl libboost-all-dev libsdl2-dev swig
RUN apt-get -y install wget unzip
RUN apt-get clean

RUN rm -rf /var/lib/apt/lists/*
RUN easy_install pip

# so we don't get SSL warnings
RUN pip install requests[security]

# OpenAI Gym
RUN pip install gym[doom]

# Dependencies for modular_rl
RUN pip install theano keras tabulate numpy scipy scikit-image matplotlib h5py

# clean pip
RUN rm -rf ~/.cache/pip

# project specific modular_rl branch
RUN git clone https://github.com/domluna/modular_rl.git \
        && cd modular_rl \
        && git checkout doms-branch

ENV PYTHONPATH /modular_rl

# project code/snapshots
COPY . ml_p5_capstone

ENTRYPOINT ["bash", "/ml_p5_capstone/setup.sh"]
