FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip wget git && \
    rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /workspace

# Only copy dependency files first (for better caching)
COPY requirements.txt requirements_no_deps.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --root-user-action=ignore -r requirements.txt
RUN pip install --root-user-action=ignore --no-deps -r requirements_no_deps.txt

CMD ["bash"]