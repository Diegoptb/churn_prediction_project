# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.7

FROM python:${PYTHON_VERSION}-slim as base

# Evita archivos .pyc y logs en buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Instalar dependencias del SISTEMA (como curl para el healthcheck)
# Esto debe hacerse como root, antes de cambiar de usuario.
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Crear un usuario no privilegiado (appuser)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    appuser

# 3. Instalación de librerías de Python
# Usamos el mount de caché para acelerar builds futuros
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# 4. Copiar el código fuente y asignar permisos AL MISMO TIEMPO
# Usamos --chown para que los archivos pertenezcan a appuser, no a root.
COPY --chown=appuser:appuser . .

# 5. Cambiar al usuario limitado
USER appuser

# Exponer el puerto estándar de Streamlit
EXPOSE 8501

# Verificación de Salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de arranque
ENTRYPOINT ["/bin/bash", "-c", "echo -e '\\n\\n  🚀 APP LISTA EN: http://localhost:8501 \\n' && streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0"]