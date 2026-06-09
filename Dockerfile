FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and ReportLab
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and replace tensorflow-macos with tensorflow for Linux
COPY requirements.txt .
RUN sed -i 's/tensorflow-macos/tensorflow-cpu/g' requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Create required artifact directories
RUN mkdir -p artifacts/uploads artifacts/demo artifacts/reports artifacts/models logs

EXPOSE 8000

CMD ["python", "app.py"]