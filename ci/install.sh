set -ex

main() {
    curl -LSfs https://japaric.github.io/trust/install.sh | \
        sh -s -- \
           --force \
           --git japaric/xargo \
           --tag v0.3.7 \
           --target x86_64-unknown-linux-musl

    rustup component add rust-src
}

main
