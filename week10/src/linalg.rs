use anyhow::{Result, ensure};
use std::ffi::c_char;

#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(not(target_os = "macos"), link(name = "lapack"), link(name = "blas"))]
unsafe extern "C" {
    fn dgesdd_(
        job: *const c_char,
        m: &i32,
        n: &i32,
        a: *mut f64,
        lda: &i32,
        s: *mut f64,
        u: *mut f64,
        ldu: &i32,
        vt: *mut f64,
        ldvt: &i32,
        work: *mut f64,
        lwork: &i32,
        iwork: *mut i32,
        info: &mut i32,
    );
    fn dgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: &i32,
        n: &i32,
        k: &i32,
        alpha: &f64,
        a: *const f64,
        lda: &i32,
        b: *const f64,
        ldb: &i32,
        beta: &f64,
        c: *mut f64,
        ldc: &i32,
    );
}

pub struct Svd {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
}

pub fn svd(mut a: Vec<f64>, rows: usize, cols: usize) -> Result<Svd> {
    // LAPACK sees the transpose of the row-major input. Its V and U become our U and V.
    let (m, n) = (i32::try_from(cols)?, i32::try_from(rows)?);
    let q = rows.min(cols);
    let mut u = vec![0.; rows * q];
    let mut vt = vec![0.; q * cols];
    let mut s = vec![0.; q];
    let mut iwork = vec![0; 8 * q];
    let mut info = 0;
    let mut query = [0.];
    let job = c"S".as_ptr();
    let ldvt = q as i32;
    ensure!(a.len() == rows * cols && q > 0, "invalid SVD input shape");
    unsafe {
        dgesdd_(
            job,
            &m,
            &n,
            a.as_mut_ptr(),
            &m,
            s.as_mut_ptr(),
            vt.as_mut_ptr(),
            &m,
            u.as_mut_ptr(),
            &ldvt,
            query.as_mut_ptr(),
            &-1,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    ensure!(info == 0, "SVD workspace query failed ({info})");
    let lwork = query[0] as i32;
    let mut work = vec![0.; lwork as usize];
    unsafe {
        dgesdd_(
            job,
            &m,
            &n,
            a.as_mut_ptr(),
            &m,
            s.as_mut_ptr(),
            vt.as_mut_ptr(),
            &m,
            u.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    ensure!(info == 0, "SVD failed ({info})");
    Ok(Svd { u, s, vt })
}

pub fn matmul(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    let (m, n, k) = (cols as i32, rows as i32, inner as i32);
    let mut c = vec![0.; rows * cols];
    let normal = c"N".as_ptr();
    // Compute (AB)^T = B^T A^T in column-major storage.
    unsafe {
        dgemm_(
            normal,
            normal,
            &m,
            &n,
            &k,
            &1.,
            b.as_ptr(),
            &m,
            a.as_ptr(),
            &k,
            &0.,
            c.as_mut_ptr(),
            &m,
        );
    }
    c
}
