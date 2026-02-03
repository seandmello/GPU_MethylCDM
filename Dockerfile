FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir \
    einops \
    einops-exts \
    pandas \
    pillow \
    tqdm \
    kornia \
    resize-right

WORKDIR /app
COPY . /app/

ENTRYPOINT ["python", "newnew_main.py"]
