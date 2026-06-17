# Grabette SLAM pipeline — HuggingFace Space (Docker SDK), self-contained.
#
# Bundles the offline_vslam binary + the grabette-postprocess package + a Gradio
# UI, so SLAM runs in-process. Docker-in-Docker is unavailable on Spaces, hence
# the binary is built into this image and called directly (binary= in run_oak_slam).
#
# Self-contained: the grabette-postprocess package (with offline_vslam.cpp) is
# vendored into the build context as ./grabette-postprocess — no git clone, no
# private-repo secret. See deploy.sh, which assembles that layout.

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# --- System deps: RTAB-Map build chain + Python 3.12 (deadsnakes) + ffmpeg ---
# Python 3.12 is required by lerobot. opencv-python (pip wheel) needs libgl1.
# gnupg + dirmngr are needed by add-apt-repository to import the PPA key
# (ubuntu:22.04 minimal ships without them).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg dirmngr \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config ffmpeg \
        libopencv-dev nlohmann-json3-dev libsqlite3-dev libeigen3-dev libpcl-dev \
        libgl1 libglib2.0-0 \
        python3.12 python3.12-venv python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Build RTAB-Map from source ---
ARG RTABMAP_VERSION=0.23.1
RUN git clone --depth 1 --branch ${RTABMAP_VERSION} https://github.com/introlab/rtabmap.git /tmp/rtabmap \
 && cmake -S /tmp/rtabmap -B /tmp/rtabmap/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_APP=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
 && cmake --build /tmp/rtabmap/build --parallel 2 \
 && cmake --install /tmp/rtabmap/build \
 && ldconfig \
 && rm -rf /tmp/rtabmap

# --- Vendored grabette-postprocess (package + offline_vslam.cpp) ---
COPY grabette-postprocess /opt/grabette-postprocess

# --- Compile offline_vslam ---
RUN RTABMAP_INCLUDE=$(find /usr/local/include -maxdepth 1 -name "rtabmap*" -type d | head -1) \
 && PCL_INCLUDE=$(find /usr/include -maxdepth 1 -name "pcl*" -type d | head -1) \
 && g++ -std=c++17 -O2 /opt/grabette-postprocess/docker/oak_vslam/offline_vslam.cpp \
        -o /usr/local/bin/offline_vslam \
        -I${RTABMAP_INCLUDE} -I${PCL_INCLUDE} $(pkg-config --cflags eigen3) \
        -lrtabmap_core -lrtabmap_utilite $(pkg-config --cflags --libs opencv4) \
 && chmod +x /usr/local/bin/offline_vslam
ENV OAK_VSLAM_BINARY=/usr/local/bin/offline_vslam

# --- Python env: grabette-postprocess (pulls lerobot) + UI deps ---
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir /opt/grabette-postprocess \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# --- Run as the non-root user Spaces expect (uid 1000), with writable caches ---
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    SYSTEM=spaces
# SYSTEM=spaces lets Gradio's get_space() recognize this Docker Space so it uses
# the real HF OAuth flow (vars injected by hf_oauth: true) instead of the local mock.
WORKDIR /home/user/app
COPY --chown=user app.py pipeline.py /home/user/app/

EXPOSE 7860
CMD ["python", "app.py"]
