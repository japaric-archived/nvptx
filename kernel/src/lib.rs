#![feature(abi_ptx)]
#![no_std]

extern crate nvptx_builtins as intrinsics;

/// Add two "vectors" of length `n`. `c <- a + b`
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add(a: *const f32,
                                      b: *const f32,
                                      c: *mut f32,
                                      n: usize) {
    let i = intrinsics::block_dim_x()
        .wrapping_mul(intrinsics::block_idx_x())
        .wrapping_add(intrinsics::thread_idx_x()) as isize;

    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

/// Copies an array of `n` floating point numbers from `src` to `dst`
#[no_mangle]
pub unsafe extern "ptx-kernel" fn memcpy(dst: *mut f32,
                                         src: *const f32,
                                         n: usize) {
    let i = (intrinsics::block_dim_x())
        .wrapping_mul(intrinsics::block_idx_x())
        .wrapping_add(intrinsics::thread_idx_x()) as isize;

    if (i as usize) < n {
        *dst.offset(i) = *src.offset(i);
    }
}

#[repr(C)]
pub struct Rgba {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rgba2gray(rgba: *const Rgba,
                                            gray: *mut u8,
                                            width: i32,
                                            height: i32) {
    let x = intrinsics::block_idx_x()
        .wrapping_mul(intrinsics::block_dim_x())
        .wrapping_add(intrinsics::thread_idx_x());
    let y = intrinsics::block_idx_y()
        .wrapping_mul(intrinsics::block_dim_y())
        .wrapping_add(intrinsics::thread_idx_y());

    if x < width && y < height {
        let i = y.wrapping_mul(width).wrapping_add(x) as isize;

        let Rgba { r, g, b, .. } = *rgba.offset(i);

        *gray.offset(i) = (0.299 * f32::from(r) + 0.589 * f32::from(g) +
                           0.114 *
                           f32::from(b)) as u8;
    }
}
