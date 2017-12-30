set -euxo pipefail

main() {
    local targets=(
        nvptx-nvidia-cuda
        nvptx64-nvidia-cuda
    )
    local toml=kernel/Cargo.toml

    for target in ${targets[@]}; do
        cargo clean --manifest-path $toml
        xargo rustc --manifest-path $toml --release --target $target -- --emit=asm
        cat $(find kernel/target/$target/release -name '*.s')
    done
}

main
