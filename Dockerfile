# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.10 slim keeps the image small while matching the target runtime.
FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 / libglib2.0 are required by OpenCV.
# ffmpeg is required by imageio-ffmpeg for H.264 encoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer separately from source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Streamlit configuration ───────────────────────────────────────────────────
# Disable the browser-open behaviour and set a fixed port for container use.
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the Streamlit port
EXPOSE 8501

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Pass the Groq API key at runtime:
#   docker run -e GROQ_API_KEY=your_key_here -p 8501:8501 sportvision
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

