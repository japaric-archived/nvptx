//! Convert a color image to grayscale

extern crate cuda;
extern crate image;

use std::ops::Add;
use std::time::{Duration, Instant};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;
use std::{env, mem};

use cuda::driver::{Any, Block, Device, Direction, Grid, Result};
use cuda::driver;
use image::{ColorType, Pixel, Rgba, RgbaImage};

fn main() {
    let mut args = env::args_os().skip(1);

    let ptx = &mut String::new();
    File::open(args.next().unwrap())
        .unwrap()
        .read_to_string(ptx)
        .unwrap();

    let img = image::open(args.next().unwrap()).unwrap().to_rgba();

    run(ptx, &img).unwrap()
}

fn run(ptx: &str, img: &RgbaImage) -> Result<()> {
    const BLOCK_SIZE: u32 = 32;

    let h_rgba = img.as_ptr();
    let (w, h) = img.dimensions();
    let npixels = (w * h) as usize;
    let nbytes = npixels * mem::size_of::<Rgba<u8>>();

    println!("Image size: {}x{} - {} pixels - {} bytes",
             w,
             h,
             npixels,
             nbytes);

    // Allocate memory on the host
    let mut h_gray = vec![0u8; npixels];

    // Initialize driver, and load kernel
    driver::initialize()?;

    let device = Device(0)?;
    let ctx = device.create_context()?;
    let module = ctx.load_module(&CString::new(ptx).unwrap())?;
    let kernel =
        module.function(&CStr::from_bytes_with_nul(b"rgba2gray\0").unwrap())?;

    println!();
    println!("RGBA -> grayscale on the GPU");
    let mut ds = vec![];

    // Allocate memory on the device
    let now = Instant::now();
    let (d_rgba, d_gray) =
        unsafe { (driver::allocate(nbytes)?, driver::allocate(npixels)?) };
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - `malloc`", elapsed);

    // Memcpy Host -> Device
    let now = Instant::now();
    unsafe {
        driver::copy(h_rgba, d_rgba, nbytes, Direction::HostToDevice)?;
    }
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - `memcpy` (CPU -> GPU)", elapsed);

    // Launch kernel
    let now = Instant::now();
    kernel.launch(&[Any(&d_rgba),
                  Any(&d_gray),
                  Any(&(w as i32)),
                  Any(&(h as i32))],
                Grid::xy((w - 1) / BLOCK_SIZE + 1, (h - 1) / BLOCK_SIZE + 1),
                Block::xy(BLOCK_SIZE, BLOCK_SIZE))?;
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - Executing the kernel", elapsed);

    // Memcpy device -> host
    let now = Instant::now();
    unsafe {
        driver::copy(d_gray,
                     h_gray.as_mut_ptr(),
                     npixels,
                     Direction::DeviceToHost)?;
    }
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - `memcpy` (GPU -> CPU)", elapsed);

    // Free memory on the device
    let now = Instant::now();
    unsafe {
        driver::deallocate(d_rgba as *mut _)?;
        driver::deallocate(d_gray as *mut _)?;
    }
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - `free`", elapsed);

    println!("    ----------------------------------------");
    println!("    {:?} - TOTAL",
             ds.iter().cloned().fold(Duration::new(0, 0), Add::add));

    image::save_buffer("gray.gpu.png", &h_gray, w, h, ColorType::Gray(8))
        .unwrap();

    println!();
    println!("RGBA -> grayscale on the CPU");
    let mut ds = vec![];

    let now = Instant::now();
    let mut gray = {
        let mut v = Vec::with_capacity(npixels);
        unsafe { v.set_len(npixels) };
        v
    };
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - `malloc`", elapsed);

    let now = Instant::now();
    for (i, p) in img.pixels().enumerate() {
        let (r, g, b, _) = p.channels4();

        gray[i] = (0.299 * f32::from(r) + 0.589 * f32::from(g) +
                   0.114 * f32::from(b)) as u8;
    }
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - conversion", elapsed);

    // Verify correctness
    assert_eq!(h_gray, gray);

    image::save_buffer("gray.cpu.png", &gray, w, h, ColorType::Gray(8))
        .unwrap();

    let now = Instant::now();
    mem::drop(gray);
    let elapsed = now.elapsed();
    ds.push(elapsed);
    println!("    {:?} - `free`", elapsed);

    println!("    ----------------------------------------");
    println!("    {:?} - TOTAL",
             ds.iter().cloned().fold(Duration::new(0, 0), Add::add));

    Ok(())
}
