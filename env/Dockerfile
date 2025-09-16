FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update && apt-get install -y \
    imagemagick \
    ffmpeg \
    zsh \
    curl \
    build-essential \
    ca-certificates \
    sudo \
    git \
    zip \
    bzip2 \
    libx11-6 \
    libgl1-mesa-glx \
    libegl1-mesa \
    wget \
    python3.10-venv \
    software-properties-common \
    ssh \
    rsync


# install requirements
RUN mkdir /workspace
WORKDIR /workspace
COPY env/requirements.txt .
COPY env/install.sh .
RUN bash install.sh


####################################################################################################
# Setup for local development

# add local user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create non root user, add it to custom group and setup environment.
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -d /home/${USERNAME} --shell /usr/bin/bash 

# OPTIONAL
# If sudo privilages are not required comment below line
# Create simple password for user and add it to sudo group
# Update group so that it is not required to type password for commands: apt update/upgrade/install/remove
RUN echo "root:password" | chpasswd \
    && echo "${USERNAME}:password" | chpasswd \
    && usermod -aG sudo ${USERNAME} \
    && echo "%sudo ALL=NOPASSWD:/usr/bin/apt-get update, /usr/bin/apt-get upgrade, /usr/bin/apt-get install, /usr/bin/apt-get remove" >> /etc/sudoers

# Create workspace folder and change ownership to new user
RUN chown ${USER_UID}:${USER_GID} /home/${USERNAME}
WORKDIR /home/${USERNAME}

# Switch to new user.
USER ${USERNAME}

# Add local user binary folder to PATH variable.
WORKDIR /home/${USERNAME}
ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"

# Run bash on container start
CMD ["bash"]