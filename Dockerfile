# Python 3.11 기반 이미지 사용
# MuJoCo 공식 Python 패키지는 Python 3.10 이상을 전제로 하므로,
# 교육 환경에서는 Python 3.11을 안정적인 기준으로 사용한다.
FROM python:3.11-slim

# 컨테이너 내부 작업 폴더
WORKDIR /workspace

# MuJoCo 실행과 렌더링에 필요한 Linux 패키지 설치
# libgl1, libegl1, libosmesa6 등은 MuJoCo 렌더링/오프스크린 렌더링 오류를 줄이기 위해 설치한다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
	terminator \
    nano \
    bash \
    build-essential \
    libgl1 \
    libegl1 \
    libgles2 \
    libosmesa6 \
    libglfw3 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python -m pip install --upgrade pip setuptools wheel

# Python 패키지 목록 복사 후 설치
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 기본 실행 셸
CMD ["bash"]