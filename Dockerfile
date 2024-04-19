FROM tensorflow/tensorflow:latest-gpu

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app
COPY --chown=user:user eye_model_vgg16_binary.keras /opt/app
COPY --chown=user:user eye_model_vgg16_multi.keras /opt/app

ENTRYPOINT ["python", "inference.py"]
