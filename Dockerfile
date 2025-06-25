FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install optional server extras
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy project
COPY . .

# Install as package (editable)
RUN pip install --no-cache-dir -e .

# Default envs
ENV SIGLA_API_KEY=""

EXPOSE 8000

# Entrypoint: start API server with provided index path env `SIGLA_INDEX`
CMD ["python", "-m", "sigla", "serve", "-s", "${SIGLA_INDEX:-capsules}", "-p", "8000"] 