//! Add two (mathematical) vectors on the GPU

extern crate cuda;
extern crate rand;

use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;
use std::{env, mem};

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

    // Allocate memory on the host
    let mut rng: XorShiftRng = rand::thread_rng().gen();

    let h_a: Vec<f32> = rng.gen_iter().take(SIZE).collect();
    let h_b: Vec<f32> = rng.gen_iter().take(SIZE).collect();
    let mut h_c: Vec<f32> = (0..SIZE).map(|_| 0.).collect();

    // Initialize driver, and load kernel
    driver::initialize()?;

    let device = Device(0)?;
    let ctx = device.create_context()?;
    let module = ctx.load_module(&CString::new(ptx).unwrap())?;
    let kernel =
        module.function(&CStr::from_bytes_with_nul(b"add\0").unwrap())?;

    // Allocate memory on the device
    let (d_a, d_b, d_c) = unsafe {
        let bytes = SIZE * mem::size_of::<f32>();

        (driver::allocate(bytes)? as *mut f32,
         driver::allocate(bytes)? as *mut f32,
         driver::allocate(bytes)? as *mut f32)
    };

    // Memcpy Host -> Device
    unsafe {
        driver::copy(h_a.as_ptr(), d_a, SIZE, Direction::HostToDevice)?;
        driver::copy(h_b.as_ptr(), d_b, SIZE, Direction::HostToDevice)?;
    }

    // Launch kernel
    let n = SIZE as u32;
    let nthreads = device.max_threads_per_block()? as u32;
    let nblocks = n / nthreads;
    kernel.launch(&[Any(&d_a), Any(&d_b), Any(&d_c), Any(&n)],
                Grid::x(nblocks),
                Block::x(nthreads))?;

    // Memcpy device -> host
    unsafe {
        driver::copy(d_c, h_c.as_mut_ptr(), SIZE, Direction::DeviceToHost)?;
    }

    // Free memory on the device
    unsafe {
        driver::deallocate(d_a as *mut _)?;
        driver::deallocate(d_b as *mut _)?;
        driver::deallocate(d_c as *mut _)?;
    }

    // Perform the same computation on the host
    let c = h_a.iter().zip(h_b).map(|(a, b)| a + b).collect::<Vec<_>>();

    // Verify correctness
    assert_eq!(c, h_c);

    Ok(())
}
