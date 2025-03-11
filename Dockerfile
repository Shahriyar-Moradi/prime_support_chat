# Use an official lightweight Python image.
FROM python:3.9-slim

# Set environment variables.


# Set the working directory.
WORKDIR /app

# Copy requirements file first to leverage Docker cache.
COPY requirements.txt .

# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Expose port 8000 for the app.
EXPOSE 8000

# Run the application with Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
