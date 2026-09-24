FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home --uid 10001 cockpit && mkdir -p /app/state && chown -R cockpit /app/state /app/data
USER cockpit
# Cloud Run, Render and Railway pass the port in $PORT; locally it defaults to 8501.
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK CMD python -c "import os,urllib.request;urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8501')+'/_stcore/health')"
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0"]
