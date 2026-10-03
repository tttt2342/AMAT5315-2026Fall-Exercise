#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};
#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 {
    x * x * x
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.0);
    let (_, adj) = cube_reverse(x, 1.0);
    unsafe {
        *out = y;
        *out.add(1) = dy;
        *out.add(2) = adj;
    }
}
// State layout: [u^(n-1), u^n]. Constants contain a=1/(1+sigma*dt),
// b=(1-sigma*dt), and the source q^n; both input and output are complete states.
#[autodiff_forward(step_jvp, Const, Const, Const, Const, Dual, Dual, Const, Dual)]
#[autodiff_reverse(
    step_vjp, Const, Const, Const, Const, Duplicated, Duplicated, Const, Duplicated
)]
fn step(nx: usize, nz: usize, dt2: f64, idx2: f64, s: &[f64], c: &[f64], k: &[f64], o: &mut [f64]) {
    let l = nx * nz;
    for i in 0..l {
        o[i] = s[l + i];
        o[l + i] = 0.0;
    }
    for z in 1..nz - 1 {
        for x in 1..nx - 1 {
            let i = z * nx + x;
            let lap = (s[l + i - 1] + s[l + i + 1] + s[l + i - nx] + s[l + i + nx]
                - 4.0 * s[l + i])
                * idx2;
            o[l + i] = (2.0 * s[l + i] - k[l + i] * s[i]
                + dt2 * (c[i] * c[i] * lap + k[2 * l + i]))
                * k[i];
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wave_step(
    nx: usize,
    nz: usize,
    dt2: f64,
    idx2: f64,
    s: *const f64,
    c: *const f64,
    k: *const f64,
    o: *mut f64,
) {
    unsafe {
        let l = nx * nz;
        step(
            nx,
            nz,
            dt2,
            idx2,
            core::slice::from_raw_parts(s, 2 * l),
            core::slice::from_raw_parts(c, l),
            core::slice::from_raw_parts(k, 3 * l),
            core::slice::from_raw_parts_mut(o, 2 * l),
        );
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wave_jvp(
    nx: usize,
    nz: usize,
    dt2: f64,
    idx2: f64,
    s: *const f64,
    ds: *const f64,
    c: *const f64,
    dc: *const f64,
    k: *const f64,
    o: *mut f64,
    do_: *mut f64,
) {
    unsafe {
        let l = nx * nz;
        step_jvp(
            nx,
            nz,
            dt2,
            idx2,
            core::slice::from_raw_parts(s, 2 * l),
            core::slice::from_raw_parts(ds, 2 * l),
            core::slice::from_raw_parts(c, l),
            core::slice::from_raw_parts(dc, l),
            core::slice::from_raw_parts(k, 3 * l),
            core::slice::from_raw_parts_mut(o, 2 * l),
            core::slice::from_raw_parts_mut(do_, 2 * l),
        );
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wave_vjp(
    nx: usize,
    nz: usize,
    dt2: f64,
    idx2: f64,
    s: *const f64,
    bs: *mut f64,
    c: *const f64,
    bc: *mut f64,
    k: *const f64,
    o: *mut f64,
    bo: *mut f64,
) {
    unsafe {
        let l = nx * nz;
        step_vjp(
            nx,
            nz,
            dt2,
            idx2,
            core::slice::from_raw_parts(s, 2 * l),
            core::slice::from_raw_parts_mut(bs, 2 * l),
            core::slice::from_raw_parts(c, l),
            core::slice::from_raw_parts_mut(bc, l),
            core::slice::from_raw_parts(k, 3 * l),
            core::slice::from_raw_parts_mut(o, 2 * l),
            core::slice::from_raw_parts_mut(bo, 2 * l),
        );
    }
}
unsafe extern "C" {
    fn abort() -> !;
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
