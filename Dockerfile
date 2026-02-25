# Step 1: Use an official Python runtime as a parent image
# Using the version specified in .python-version for consistency
FROM python:3.12-slim

# Step 2: Set environment variables
# ENV PYTHONUNBUFFERED=1 ensures that python output is sent straight to the terminal
# ENV GR_SERVER_NAME="0.0.0.0" makes Gradio listen on all network interfaces
ENV PYTHONUNBUFFERED=1
ENV GR_SERVER_NAME="0.0.0.0"

# Step 3: Set the working directory in the container
WORKDIR /app

# Step 4: Install uv, our package manager
# We do this in a separate layer that can be cached.
RUN pip install uv

# Step 5: Copy dependency definition files
COPY pyproject.toml uv.lock* .python-version* ./

# Step 6: Install project dependencies using uv
# This leverages the uv.lock file for fast, reproducible installs.
RUN uv sync --no-cache

# Step 7: Copy the application source code
COPY src/ ./src/
COPY datasets/ ./datasets/
COPY main.py .

# NOTE: The vector store is NO LONGER pre-built in the Docker image.
# Instead, it will be created and persisted inside the container's filesystem
# on the first run, using the API key provided to the running container.

# Step 8: Expose the port Gradio will run on
EXPOSE 7860

# Step 9: Define the command to run the application
# This command starts the Gradio app when the container launches.
CMD ["uv", "run", "python", "main.py"]
