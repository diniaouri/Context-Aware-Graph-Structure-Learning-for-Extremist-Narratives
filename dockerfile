# Use Python 3.8 image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /projet_recherche_arenas

RUN apt-get update && apt-get install -y gcc g++ libgomp1
# Copy the Python script
COPY . .  
# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the Python script
# CMD ["python", "main.py"]
ENTRYPOINT ["python", "main.py"]
CMD []