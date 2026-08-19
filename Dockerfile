FROM python:3.11-slim

# LightGBM은 OpenMP 런타임(libgomp) 필요
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사해서 레이어 캐시 활용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드만 복사
# data/ 와 mlruns/ 는 런타임에 볼륨으로 마운트
COPY src/ ./src/

# 파이썬 로그가 버퍼링 없이 바로 보이게
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# MLflow 로컬 파일 백엔드
# 6주차에 서버 분리하면 이 값만 바꾸면 됨
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

CMD ["sh", "-c", "python src/preprocess.py && python src/train.py && python src/evaluate.py"]