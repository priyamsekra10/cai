FROM python:3.10-bullseye

# Set working directory first
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    default-libmysqlclient-dev \
    build-essential \
    gcc \
    g++ \
    cmake \
    python3-dev \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all application files
COPY . .

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["python3", "server.py"]





# # FROM python:3.10-bullseye


# # RUN mkdir /app
# # RUN mkdir /app/assets
# # RUN mkdir /app/utils
# # COPY *.py /app/
# # COPY requirements.txt /app/
# # copy assets/* /app/assets/
# # copy utils/* /app/utils/

# # WORKDIR /app
# # RUN pip3 install -r requirements.txt

# # EXPOSE 7860

# # CMD ["python3", "server.py"]


# # Dockerfile
# FROM python:3.10-bullseye

# # Set working directory
# # WORKDIR /app

# # Install system dependencies including MySQL client
# # RUN apt-get update && apt-get install -y \
# #     default-libmysqlclient-dev \
# #     gcc \
# #     pkg-config \
# #     && rm -rf /var/lib/apt/lists/*
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#     default-libmysqlclient-dev \
#     build-essential \
#     gcc \
#     g++ \
#     cmake \
#     python3-dev \
#     git \
#     pkg-config \
#     && rm -rf /var/lib/apt/lists/*


# # Copy requirements file
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# # COPY ./app /app/
# COPY server.py /app/
# WORKDIR /app

# # Expose port
# EXPOSE 7860

# # Command to run the application
# CMD ["python3","server.py"]


