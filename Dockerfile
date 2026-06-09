FROM python:3.10-slim

WORKDIR /app

# Force rebuild - v2

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sed -i 's/tensorflow-macos/tensorflow-cpu/g' requirements.txt

RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts/uploads artifacts/demo artifacts/reports artifacts/models logs

EXPOSE 8000

CMD ["python", "app.py"]