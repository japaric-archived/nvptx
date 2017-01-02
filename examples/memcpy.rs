//! `memcpy` on the GPU

extern crate cuda;
extern crate rand;

use std::{env, mem};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;

use cuda::driver::{self, Any, Block, Device, Direction, Grid, Result};
use rand::{Rng, XorShiftRng};

fn main() {
    let ptx = &mut String::new();

    File::open(env::args_os().skip(1).next().unwrap())
        .unwrap()
        .read_to_string(ptx)
        .unwrap();

    run(ptx).unwrap();
}

fn run(ptx: &str) -> Result<()> {
    const SIZE: usize = 1024 * 1024;

    // Allocate memory on host
    let mut rng: XorShiftRng = rand::thread_rng().gen();

    let h_a = (0..SIZE).map(|_| rng.gen()).collect::<Vec<f32>>();
    let mut h_b = (0..SIZE).map(|_| 0.).collect::<Vec<f32>>();

    // Initialize driver, and load kernel
    driver::initialize()?;

    let device = Device(0)?;
    let ctx = device.create_context()?;
    let module = ctx.load_module(&CString::new(ptx).unwrap())?;
    let kernel =
        module.function(&CStr::from_bytes_with_nul(b"memcpy\0").unwrap())?;

    // Allocate memory on device
    let (d_a, d_b) = unsafe {
        let bytes = SIZE * mem::size_of::<f32>();

        (driver::allocate(bytes)? as *mut f32,
         driver::allocate(bytes)? as *mut f32)
    };

    // Memcpy Host -> Device
    unsafe {
        driver::copy(h_a.as_ptr(), d_a, SIZE, Direction::HostToDevice)?;
    }

    // Launch kernel
    let n = SIZE as u32;
    let nthreads = device.max_threads_per_block()? as u32;
    let nblocks = (n - 1) / nthreads + 1;
    kernel.launch(&[Any(&d_b), Any(&d_a), Any(&n)],
                Grid::x(nblocks),
                Block::x(nthreads))?;

    // Memcpy device -> host
    unsafe {
        driver::copy(d_b, h_b.as_mut_ptr(), SIZE, Direction::DeviceToHost)?;
    }

    // Free memory on device
    unsafe {
        driver::deallocate(d_a as *mut _)?;
        driver::deallocate(d_b as *mut _)?;
    }

    // Verify correctness
    assert_eq!(h_a, h_b);

    Ok(())
}
