FROM python:3.11-slim

# libgomp1: LightGBM이 의존하는 OpenMP 런타임
# git: 없으면 MLflow가 mlflow.source.git.commit 태그를 기록하지 못한다 (5주차 미기록 원인)
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# .git은 이미지에 굽지 않고 compose에서 읽기 전용으로 마운트한다.
# 바인드 마운트된 .git의 소유자 uid가 컨테이너 사용자와 달라도 git이 거부하지 않게 한다.
RUN git config --global --add safe.directory /app

# 의존성 먼저 복사해서 레이어 캐시 활용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드만 복사
# data/ 와 .git 은 런타임에 볼륨으로 마운트
COPY src/ ./src/

# 파이썬 로그가 버퍼링 없이 바로 보이게
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# MLFLOW_TRACKING_URI는 여기서 기본값을 주지 않는다.
# file 백엔드로 조용히 떨어지면 모델 레지스트리 등록이 실패하므로,
# compose가 tracking server 주소를 명시적으로 주입한다.

CMD ["sh", "-c", "python src/preprocess.py && python src/train.py && python src/evaluate.py"]