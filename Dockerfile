# ----------------------------
# Base image
# ----------------------------
FROM python:3.12-slim

# ----------------------------
# Metadata
# ----------------------------
LABEL maintainer="Md Abdul Bari <abdulbari80@gmail.com>"
LABEL version="1.0"
LABEL description="End-to-end ML project with Flask web app deployed on Azure"

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy dependency files first
# ----------------------------
COPY requirements.txt setup.py ./

# ----------------------------
# Install dependencies
# ----------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# ----------------------------
# Copy project code
# ----------------------------
COPY . .

# ----------------------------
# Expose port
# ----------------------------
EXPOSE 80

# ----------------------------
# Environment variables
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ----------------------------
# Run unit tests (optional, can be skipped in prod)
# ----------------------------
# RUN python -m unittest discover -s tests

# ----------------------------
# Command to run the app
# ----------------------------
CMD ["python", "app.py"]