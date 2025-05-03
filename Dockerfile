FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories if they don't exist
RUN mkdir -p static/css static/js templates

# Make sure the app can write to the data file
RUN touch freelancer_notes.json && chmod 666 freelancer_notes.json


ENV PORT=5000
ENV HOST=0.0.0.0

# Run the application
CMD ["uvicorn", "main:app", "--host", "${HOST}", "--port", "${PORT}", "--reload"]