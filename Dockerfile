FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update && apt-get install -y git
RUN pip install uv
RUN uv venv comfyui
ENV PATH="/comfyui/bin:$PATH"
RUN pip install comfy-cli
RUN comfy --skip-prompt --no-enable-telemetry install --nvidia
COPY ./ImageEdit/custom_nodes_list.json ./deps.json
RUN comfy node install-deps --deps=deps.json
COPY ./ImageEdit/extra_model_paths.yaml /root/comfyui/ComfyUI
COPY ./ImageEdit/app /app
RUN uv venv app
ENV PATH="/app/bin:$PATH"
RUN pip install -r /app/requirements.txt