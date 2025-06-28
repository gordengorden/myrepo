# myrepo

## Initial Setup
## Running with Docker

This project provides a Docker setup for the ASR (Automatic Speech Recognition) API service. The service is built using Python and PyTorch and exposes port 8001.

### Build and Run

Use Docker Compose from root to build and run the service:

```bash
docker compose up --build
```

To stop the service:

```bash
docker-compose down
```

This will build the `asr-api` service from the `./asr` directory using the provided Dockerfile and start the API on port 8001.

### Service Details
- **Service name:** asr-api
- **Base image:** python:3.9
- **Exposed port:** 8001 (host: 8001 → container: 8001)
- **Dependencies:** Installed from `./requirements.txt`

### Configuration
- No environment variables are required by default.
- No volumes or external dependencies are needed for this service.

### API Endpoints
| Method | Route            | Description                                   |
| ------ | ---------------- | --------------------------------------------- |
| GET    | `/ping`          | Returns 'pong'; Health check                  |
| POST   | `/asr`           | Gets transcription and duration of audio file |


# ASR Train

cv-train-2a was ran on Google Colab for compute, some file paths might need to be changed to run from this root

## Package Installation
Packages used for the notebooks are found in environment.yml

## Load Model
For loading wav2vec2-large-960h-cv, you can load using
```python
model = Wav2Vec2ForCTC.from_pretrained("gordengorden/wav2vec2-large-960h-cv",)
processor = Wav2Vec2Processor.from_pretrained("gordengorden/wav2vec2-large-960h-cv")
```