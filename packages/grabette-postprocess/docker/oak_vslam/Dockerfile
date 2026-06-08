FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        libopencv-dev \
        nlohmann-json3-dev \
        libsqlite3-dev \
        libeigen3-dev \
        libpcl-dev \
        pkg-config \
 && rm -rf /var/lib/apt/lists/*

ARG RTABMAP_VERSION=0.23.1
RUN git clone --depth 1 --branch ${RTABMAP_VERSION} \
        https://github.com/introlab/rtabmap.git /tmp/rtabmap \
 && cmake -S /tmp/rtabmap -B /tmp/rtabmap/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_APP=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TOOLS=OFF \
 && cmake --build /tmp/rtabmap/build --parallel $(nproc) \
 && cmake --install /tmp/rtabmap/build \
 && ldconfig \
 && rm -rf /tmp/rtabmap

COPY offline_vslam.cpp /build/

RUN RTABMAP_INCLUDE=$(find /usr/local/include -maxdepth 1 -name "rtabmap*" -type d | head -1) \
 && PCL_INCLUDE=$(find /usr/include -maxdepth 1 -name "pcl*" -type d | head -1) \
 && g++ -std=c++17 -O2 /build/offline_vslam.cpp \
        -o /usr/local/bin/offline_vslam \
        -I${RTABMAP_INCLUDE} \
        -I${PCL_INCLUDE} \
        $(pkg-config --cflags eigen3) \
        -lrtabmap_core -lrtabmap_utilite \
        $(pkg-config --cflags --libs opencv4) \
 && chmod +x /usr/local/bin/offline_vslam

ENTRYPOINT ["/usr/local/bin/offline_vslam"]
