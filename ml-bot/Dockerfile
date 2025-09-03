FROM python:3.13-slim
WORKDIR /app
COPY chroma_langchain_db .
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app"]