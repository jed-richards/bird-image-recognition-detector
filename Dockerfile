FROM python:3.11

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . /app

# Expose port
EXPOSE 5000

# Command to run the Flask application
#CMD ["python", "your_flask_app.py"]
CMD ["flask", "--app", "bird.app.py", "run"]
