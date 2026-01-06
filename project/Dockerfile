FROM python:3.11-slim-bookworm
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG ENV=prod

COPY requirements-${ENV}.txt requirements.txt

# Upgrade pip tools, install uv (needed for lock export if desired), then
# install requirements. For prod builds, use the PyTorch CPU index so
# the +cpu wheels (torch, torchvision) can be resolved.
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install uv

RUN if [ "$ENV" = "prod" ] ; then \
      python -m pip install --upgrade pip setuptools wheel && \
      python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1+cpu torchvision==0.24.1+cpu --extra-index-url https://pypi.org/simple ; \
    fi
COPY ./app/ ./app/


CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]