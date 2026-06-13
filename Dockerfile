# 1. 기반 이미지 설정
FROM rocker/tidyverse:4.4.0

# 2. 시스템 의존성 설치
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    imagemagick \
    libmagick++-dev \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Miniforge 설치
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

# 4. Conda 경로 설정 및 환경 생성
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN conda create -y -n r-reticulate -c conda-forge --override-channels \
    python=3.10 \
    numpy \
    pandas \
    matplotlib \
    polars \
    statsmodels \
    notebook \
    jupyter_client \
    ipykernel && \
    conda clean -afy
ENV PATH=${CONDA_DIR}/envs/r-reticulate/bin:${CONDA_DIR}/bin:${PATH}

# 5. R 패키지 설치
RUN R -e "install.packages(c('reticulate', 'knitr', 'rmarkdown', 'NHANES', 'broom', 'forcats', 'IRkernel'))" && \
    R -e "IRkernel::installspec(user = FALSE)"

# 6. reticulate가 사용할 Python 경로 고정
ENV RETICULATE_PYTHON=/opt/conda/envs/r-reticulate/bin/python

# 7. Binder용 사용자 및 노트북 배치
ENV NB_USER=jovyan
ENV NB_UID=1000
RUN usermod -l ${NB_USER} rstudio && \
    usermod -d /home/${NB_USER} -m ${NB_USER} && \
    chown -R ${NB_USER}:users /opt/conda /home/${NB_USER}

COPY _site/hw03.ipynb /home/${NB_USER}/hw03.ipynb
RUN chown ${NB_USER}:users /home/${NB_USER}/hw03.ipynb

USER ${NB_USER}
WORKDIR /home/${NB_USER}

EXPOSE 8888
