# syntax=docker/dockerfile:1
ARG RUST_VERSION=1.95.0

# Optional development tools; graphical execution still runs natively by default.
FROM rust:${RUST_VERSION}-bookworm AS development
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang \
        lld \
        pkg-config \
        libasound2-dev \
        libudev-dev \
        libwayland-dev \
        libx11-dev \
        libxkbcommon-dev \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*
RUN rustup component add rustfmt clippy
WORKDIR /workspace
CMD ["bash"]

# Bevy CLI and wasm-bindgen are installed once, before copying game sources.
FROM rust:${RUST_VERSION}-bookworm AS web-tools
RUN rustup target add wasm32-unknown-unknown
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/tmp/cargo-install-target \
    CARGO_TARGET_DIR=/tmp/cargo-install-target cargo install --locked \
        --git https://github.com/TheBevyFlock/bevy_cli \
        --rev 53fea37954e71b872df815ad81a3809d620db856 \
        --no-default-features --features web bevy_cli \
    && CARGO_TARGET_DIR=/tmp/cargo-install-target cargo install --locked \
        --version 0.2.128 wasm-bindgen-cli

FROM web-tools AS web-build
WORKDIR /build
COPY . .
# wasm-bindgen-cli must match the wasm-bindgen version in Cargo.lock exactly.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    bevy build --locked --release web --bundle --wasm-opt false \
    && cp -r target/bevy_web/web-release/dodge-royale /web-dist

# Static browser deployment: PostgreSQL and native workers stay on the server.
FROM nginx:1.30.4-alpine AS web
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY --from=web-build /web-dist /usr/share/nginx/html
USER nginx
EXPOSE 8080
ENTRYPOINT ["nginx"]
CMD ["-g", "daemon off;"]

FROM rust:${RUST_VERSION}-bookworm AS build
WORKDIR /build
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked --no-default-features \
    && cp target/release/dodge-royale /usr/local/bin/dodge-royale

# The container runs Bevy headlessly, so no display server or GPU is required.
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 dodgeroyale \
    && useradd --uid 10001 --gid dodgeroyale --no-create-home \
        --shell /usr/sbin/nologin dodgeroyale
COPY --from=build /usr/local/bin/dodge-royale /usr/local/bin/dodge-royale
USER 10001:10001
ENTRYPOINT ["dodge-royale"]
