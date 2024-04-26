FROM docker.io/rust:slim as build

COPY src /build/src
COPY Cargo.toml /build/Cargo.toml
COPY Cargo.lock /build/Cargo.lock

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-program-options-dev \
    libgmp-dev \
    libhwloc-dev \
    libtbb-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mt-kahypar

RUN git clone --recursive https://github.com/kahypar/mt-kahypar.git .
RUN cmake -G Ninja -B build -D CMAKE_INSTALL_PREFIX=/usr -D MT_KAHYPAR_DISABLE_BOOST=true && \
    cmake --build build --target mtkahypar && \
    cmake --install build

WORKDIR /build

RUN cargo build --release

FROM docker.io/debian:stable-slim

LABEL org.opencontainers.image.source=https://github.com/SoftVarE-Group/d-dnnf-reasoner
LABEL org.opencontainers.image.description="A d-DNNF reasoner"
LABEL org.opencontainers.image.licenses=LGPL-3.0-or-later

RUN apt-get update && apt-get install -y \
    libgmp10 \
    libhwloc15 \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/lib/x86_64-linux-gnu/libmtkahypar.so /usr/lib/libmtkahypar.so
COPY --from=build /build/target/release/ddnnife /usr/bin/ddnnife
COPY --from=build /build/target/release/dhone /usr/bin/dhone

ENTRYPOINT ["/usr/bin/ddnnife"]
