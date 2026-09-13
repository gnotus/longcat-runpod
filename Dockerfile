FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/longcat
RUN git init upstream && cd upstream && git remote add origin https://github.com/meituan-longcat/LongCat-Video.git && git fetch --depth 1 origin 6b3f4b8582a8bc3f20f795735f5383716c4ba794 && git checkout --detach FETCH_HEAD
COPY requirements.txt install_flash.py ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python install_flash.py
ENV PYTHONPATH=/opt/longcat/upstream PYTHONUNBUFFERED=1 HF_HOME=/runpod-volume/longcat-cache HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
COPY contract.py worker.py verify_imports.py ./
RUN python verify_imports.py
ENTRYPOINT ["python","-u","/opt/longcat/worker.py"]
