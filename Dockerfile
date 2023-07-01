FROM ubuntu:latest as builder
# We NEED that line!
# Otherwise, cargo has a tantrum, finds an infinite amount of file system loops, and hangs itself :'D
WORKDIR /reasoner

# Update default packages
RUN apt-get -qq update

# packages for later installing cargo
RUN apt-get install -y -q build-essential curl

# gmp
RUN apt-get install -y -q libgmp-dev m4 diffutils gcc make

# d4v2
RUN apt-get install -y -q ninja-build git libz-dev cmake libboost-all-dev

# Get Rust; NOTE: using sh for better compatibility with other base images
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Add .cargo/bin to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

COPY ./ ./

# Build d4v2 in an extra step outside of the build script to allow for better caching
# while building the image. Running the build script of d4v2 is very labor-intensive.
RUN mkdir d4v2
RUN git clone https://github.com/SoftVarE-Group/d4v2.git d4v2/.
WORKDIR /reasoner/d4v2
RUN ./build.sh

RUN cargo build --release
WORKDIR /reasoner
RUN cp ./target/release/ddnnife ./

# Move the binary in an enviroment that is smaller in size, saving around 2/3 of the size
FROM frolvlad/alpine-glibc
COPY --from=builder /reasoner/ddnnife ./ddnnife
RUN apk add gmp-dev
ENTRYPOINT ["./ddnnife"]