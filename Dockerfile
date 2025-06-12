# We use miniconda3 as the base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker layer cache
COPY requirements.txt .

# Create conda environment and install dependencies
RUN conda create -n kernel-bench python=3.10 -y && \
    echo "source activate kernel-bench" >> ~/.bashrc && \
    /bin/bash -c "source activate kernel-bench && pip install --no-cache-dir -r requirements.txt"

# Copy the entire project
COPY . .

# Install the project in the conda environment
RUN /bin/bash -c "source activate kernel-bench && pip install -e ."

# Set environment variable to ensure correct Python is used
ENV PATH="/opt/conda/envs/kernel-bench/bin:$PATH"

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate conda environment when starting
ENTRYPOINT ["conda", "run", "-n", "kernel-bench"]
CMD ["python"]