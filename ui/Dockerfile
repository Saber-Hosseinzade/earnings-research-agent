# The UI container: Streamlit behind the ECS Express Mode load balancer (infra/ui_stack.py). Built by CodeBuild
# for ARM64 (infra/ui_image_stack.py, scripts/deploy_ui.sh), so no local Docker is needed.
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Pinned by uv.lock (requirements.txt is generated from it).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py view.py runtime_client.py entrypoint.py ./
RUN useradd --create-home --uid 10001 app && mkdir -p /app/.streamlit && chown app /app/.streamlit
USER app

EXPOSE 8501
CMD ["python", "entrypoint.py"]
