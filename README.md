# Status

This documentation about an unstable feature is **UNMAINTAINED** and was written
over a year ago. Things may have drastically changed since then; read this at
your own risk! If you are interested in modern Rust on GPU development check out
https://github.com/rust-cuda/wg

-- @japaric, 2018-12-08

---

# `nvptx`

> How to: Run Rust code on your NVIDIA GPU

* [First steps](#first-steps)
  * [Targets](#targets)
  * [Minimal example](#minimal-example)
  * [Global functions](#global-functions)
  * [Avoiding mangling](#avoiding-mangling)
  * [Optimization](#optimization)
* [Examples](#examples)
* [Problems?](#problems)
* [License](#license)
    * [Contribution](#contribution)

## First steps

Since 2016-12-31, `rustc` can compile Rust code to PTX (Parallel Thread
Execution) code, which is like GPU assembly, via `--emit=asm` and the right
`--target` argument. This PTX code can then be loaded and executed on a GPU.

*However*, a few days later 128-bit integer support landed in rustc and
broke compilation of the `core` crate for NVPTX targets (LLVM assertions).
Furthermore, there was no nightly release between these two events so it was not
possible to use the NVPTX backend with a nightly compiler.

Just recently (2017-05-18) I realized (thanks to [this blog post]) that we can
work around the problem by compiling a *fork* of the core crate that doesn't
contain code that involves 128-bit integers. Which is a bit unfortunate but,
hey, if it works then it works.

[this blog post]: https://gergo.erdi.hu/blog/2017-05-12-rust_on_avr__beyond_blinking/

### Targets

The required targets are not built into the compiler (they are not in `rustc
--print target-list`) but are available as JSON files in this repository:

- [`nvptx64-nvidia-cuda.json`](nvptx64-nvidia-cuda.json), 64-bit PTX, and
- [`nvptx-nvidia-cuda.json`](nvptx-nvidia-cuda.json), 32-bit PTX

If the host is running a 64-bit OS, you should use the nvptx64 target.
Otherwise, use the nvptx target.

### Minimal example

Here's a minimal example of emitting PTX from a Rust crate:

```
$ cargo new --lib kernel && cd $_

$ cat src/lib.rs
```

``` rust
#![no_std]

fn foo() {}
```

```
# emitting debuginfo is not supported for the nvptx targets
$ edit Cargo.toml && tail -n2 $_
[profile.dev]
debug = false

# The JSON file must be in the current directory
$ test -f nvptx64-nvidia-cuda.json && echo OK
OK

# You'll need to use Xargo to build the `core` crate "on the fly"
# Install it if you don't already have it
$ cargo install xargo || true

# Then instruct Xargo to compile a fork of the core crate that contains no
# 128-bit integers
$ edit Xargo.toml && cat Xargo.toml
[dependencies.core]
git = "https://github.com/japaric/core64"

# Xargo has the exact same CLI as Cargo
$ xargo rustc --target nvptx64-nvidia-cuda -- --emit=asm
   Compiling core v0.0.0 (file://$SYSROOT/lib/rustlib/src/rust/src/libcore)
    Finished release [optimized] target(s) in 18.74 secs
   Compiling kernel v0.1.0 (file://$PWD)
    Finished debug [unoptimized] target(s) in 0.4 secs
```

The PTX code will be available as a `.s` file in the `target` directory:

```
$ find -name '*.s'
./target/nvptx64-nvidia-cuda/debug/deps/kernel-e916cff045dc0eeb.s

$ cat $(find -name '*.s')
.version 3.2
.target sm_20
.address_size 64

.func _ZN6kernel3foo17h24d36fb5248f789aE()
{
        .local .align 8 .b8     __local_depot0[8];
        .reg .b64       %SP;
        .reg .b64       %SPL;

        mov.u64         %SPL, __local_depot0;
        bra.uni         LBB0_1;
LBB0_1:
        ret;
}
```

### Global functions

Although this PTX module (the whole file) can be loaded on the GPU, the function
`foo` contained in it *can't* be "launched" by the host because it's a *device*
function. Only *global* functions (AKA kernels) can be launched by the hosts.

To turn `foo` into a global function, its ABI must be changed to "ptx-kernel":

``` rust
#![feature(abi_ptx)]
#![no_std]

extern "ptx-kernel" fn foo() {}
```

With that change the PTX of the `foo` function will now look like this:

```
.entry _ZN6kernel3foo17h24d36fb5248f789aE()
{
        .local .align 8 .b8     __local_depot0[8];
        .reg .b64       %SP;
        .reg .b64       %SPL;

        mov.u64         %SPL, __local_depot0;
        bra.uni         LBB0_1;
LBB0_1:
        ret;
}
```

`foo` is now a global function because it has the `.entry` directive instead of
the `.func` one.

### Avoiding mangling

With the CUDA API, one can retrieve functions from a PTX module by their name.
`foo`'s' final name in the PTX module has been mangled and looks like this:
`_ZN6kernel3foo17h24d36fb5248f789aE`.

To avoid mangling the `foo` function add the `#[no_mangle]` attribute to it.

``` rust
#![feature(abi_ptx)]
#![no_std]

#[no_mangle]
extern "ptx-kernel" fn foo() {}
```

This will result in the following PTX code:

```
.entry foo()
{
        .local .align 8 .b8     __local_depot0[8];
        .reg .b64       %SP;
        .reg .b64       %SPL;

        mov.u64         %SPL, __local_depot0;
        bra.uni         LBB0_1;
LBB0_1:
        ret;
}
```

With this change you can now refer to the `foo` function using the `"foo"`
(C) string from within the CUDA API.

### Optimization

So far we have been compiling the crate using the (default) "debug" profile
which normally results in debuggable but slow code. Given that we can't emit
debuginfo when using the nvptx targets, it makes more sense to build the crate
using the "release" profile.

The catch is that we'll have to mark global functions as `pub`lic otherwise the
compiler will "optimize them away" and they won't make it into the final PTX
file.

``` rust
#![feature(abi_ptx)]
#![no_std]

#[no_mangle]
pub extern "ptx-kernel" fn foo() {}
```

```
$ cargo clean

$ xargo rustc --release --target nvptx64-nvidia-cuda -- --emit=asm

$ cat $(find -name '*.s')
.visible .entry foo()
{
        ret;
}
```

## Examples

This repository contains runnable examples of executing Rust code on the GPU.
Note that no effort has gone into ergonomically integrating both the device code
and the host code :-).

There's a [`kernel`](kernel) directory, which is a Cargo project as well, that
contains Rust code that's meant to be executed on the GPU. That's the "device"
code.

You can convert that Rust code into a PTX module using the following command:

```
$ xargo rustc \
    --manifest-path kernel/Cargo.toml \
    --release \
    --target nvptx64-nvidia-cuda \
    -- --emit=asm
```

The PTX file will available in the `kernel/target` directory.

```
$ find kernel/target -name '*.s'
kernel/target/nvptx64-nvidia-cuda/release/deps/kernel-bb52137592af9c8c.s
```

The [`examples`](examples) directory contains the "host" code. Inside that
directory, there are 3 file; each file is an example program:

- `add` - Add two (mathematical) vectors on the GPU
- `memcpy` - `memcpy` on the GPU
- `rgba2gray` - Convert a color image to grayscale

Each example program expects as first argument the path to the PTX file we
generated previously. You can run each example with a command like this:

```
$ cargo run --example add -- $(find kernel/target -name '*.s')
```

The `rgba2gray` example additionally expects a second argument: the path to the
image that will be converted to grayscale. That example also compares the
runtime of converting the image on the GPU vs the runtime of converting the
image on the CPU. Be sure to run that example in release mode to get a fair
comparison!

```
$ cargo run --release --example rgba2gray -- $(find kernel/target -name '*.s') ferris.png
Image size: 1200x800 - 960000 pixels - 3840000 bytes

RGBA -> grayscale on the GPU
    Duration { secs: 0, nanos: 602024 } - `malloc`
    Duration { secs: 0, nanos: 718481 } - `memcpy` (CPU -> GPU)
    Duration { secs: 0, nanos: 1278006 } - Executing the kernel
    Duration { secs: 0, nanos: 306315 } - `memcpy` (GPU -> CPU)
    Duration { secs: 0, nanos: 322648 } - `free`
    ----------------------------------------
    Duration { secs: 0, nanos: 3227474 } - TOTAL

RGBA -> grayscale on the CPU
    Duration { secs: 0, nanos: 12299 } - `malloc`
    Duration { secs: 0, nanos: 4171570 } - conversion
    Duration { secs: 0, nanos: 493 } - `free`
    ----------------------------------------
    Duration { secs: 0, nanos: 4184362 } - TOTAL
```

## Problems?

If you encounter any problem with the Rust -> PTX feature in the compiler,
report it to [this meta issue].

[this meta issue]: https://github.com/rust-lang/rust/issues/38789

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
