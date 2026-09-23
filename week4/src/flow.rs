use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;
use std::sync::Arc;

/// Velocity field exchanged between the `field` and `fluid` commands.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InputField {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[usize; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

/// Real-space fields recovered from a vorticity state.
#[derive(Clone, Debug)]
pub struct FlowFields {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub omega: Vec<f64>,
}

/// Fourier operations on a square periodic `[0, 2 pi)^2` grid.
pub struct SpectralGrid {
    n: usize,
    cutoff: isize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl SpectralGrid {
    pub fn new(n: usize) -> Self {
        assert!(
            n > 0 && n.is_multiple_of(2),
            "spectral grid size must be positive and even"
        );
        let mut planner = FftPlanner::new();
        Self {
            n,
            cutoff: (n / 3) as isize,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn cutoff(&self) -> usize {
        self.cutoff as usize
    }

    pub fn wave_number(&self, index: usize) -> isize {
        assert!(index < self.n);
        if index < self.n / 2 {
            index as isize
        } else {
            index as isize - self.n as isize
        }
    }

    pub fn mode_index(&self, wave_number: isize) -> usize {
        assert!(wave_number.unsigned_abs() <= self.n / 2);
        wave_number.rem_euclid(self.n as isize) as usize
    }

    fn transform_2d(&self, values: &mut [Complex<f64>], transform: &Arc<dyn Fft<f64>>) {
        assert_eq!(values.len(), self.n * self.n);
        for row in values.chunks_exact_mut(self.n) {
            transform.process(row);
        }

        let mut column = vec![Complex::new(0.0, 0.0); self.n];
        for x in 0..self.n {
            for y in 0..self.n {
                column[y] = values[y * self.n + x];
            }
            transform.process(&mut column);
            for y in 0..self.n {
                values[y * self.n + x] = column[y];
            }
        }
    }

    pub fn forward_real(&self, values: &[f64]) -> Vec<Complex<f64>> {
        assert_eq!(values.len(), self.n * self.n);
        let mut modes: Vec<Complex<f64>> = values
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        self.transform_2d(&mut modes, &self.forward);
        modes
    }

    pub fn inverse_real(&self, modes: &[Complex<f64>]) -> Vec<f64> {
        assert_eq!(modes.len(), self.n * self.n);
        let mut values = modes.to_vec();
        self.transform_2d(&mut values, &self.inverse);
        let normalization = 1.0 / (self.n * self.n) as f64;
        values
            .into_iter()
            .map(|value| normalization * value.re)
            .collect()
    }

    pub fn filter_modes(&self, modes: &mut [Complex<f64>]) {
        assert_eq!(modes.len(), self.n * self.n);
        for y in 0..self.n {
            let ky = self.wave_number(y);
            for x in 0..self.n {
                let kx = self.wave_number(x);
                if kx.abs() > self.cutoff || ky.abs() > self.cutoff {
                    modes[y * self.n + x] = Complex::new(0.0, 0.0);
                }
            }
        }
    }

    fn derivative_factor(&self, index: usize, order: usize) -> Complex<f64> {
        if order == 0 {
            return Complex::new(1.0, 0.0);
        }
        if index == self.n / 2 && order % 2 == 1 {
            return Complex::new(0.0, 0.0);
        }
        Complex::new(0.0, self.wave_number(index) as f64).powu(order as u32)
    }

    /// Differentiate a real grid field spectrally without applying dealiasing.
    pub fn derivative(&self, values: &[f64], order_x: usize, order_y: usize) -> Vec<f64> {
        let mut modes = self.forward_real(values);
        for y in 0..self.n {
            let y_factor = self.derivative_factor(y, order_y);
            for x in 0..self.n {
                modes[y * self.n + x] *= self.derivative_factor(x, order_x) * y_factor;
            }
        }
        self.inverse_real(&modes)
    }

    pub fn laplacian(&self, values: &[f64]) -> Vec<f64> {
        let mut modes = self.forward_real(values);
        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                modes[y * self.n + x] *= -(kx * kx + ky * ky);
            }
        }
        self.inverse_real(&modes)
    }

    fn velocity_modes(
        &self,
        omega_modes: &[Complex<f64>],
    ) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
        let mut u_modes = vec![Complex::new(0.0, 0.0); self.n * self.n];
        let mut v_modes = vec![Complex::new(0.0, 0.0); self.n * self.n];
        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let index = y * self.n + x;
                let squared_wave_number = kx * kx + ky * ky;
                if squared_wave_number == 0.0 {
                    continue;
                }
                let stream = omega_modes[index] / squared_wave_number;
                u_modes[index] = Complex::new(0.0, ky) * stream;
                v_modes[index] = Complex::new(0.0, -kx) * stream;
            }
        }
        (u_modes, v_modes)
    }

    pub fn velocity_from_vorticity_modes(
        &self,
        omega_modes: &[Complex<f64>],
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(omega_modes.len(), self.n * self.n);
        let mut filtered = omega_modes.to_vec();
        self.filter_modes(&mut filtered);
        let (u_modes, v_modes) = self.velocity_modes(&filtered);
        (self.inverse_real(&u_modes), self.inverse_real(&v_modes))
    }

    pub fn vorticity_from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n * self.n);
        assert_eq!(v.len(), self.n * self.n);
        let u_modes = self.forward_real(u);
        let v_modes = self.forward_real(v);
        let mut omega_modes = vec![Complex::new(0.0, 0.0); self.n * self.n];
        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let index = y * self.n + x;
                omega_modes[index] =
                    Complex::new(0.0, kx) * v_modes[index] - Complex::new(0.0, ky) * u_modes[index];
            }
        }
        self.filter_modes(&mut omega_modes);
        self.inverse_real(&omega_modes)
    }
}

/// Pseudospectral rate for the two-dimensional vorticity equation.
pub struct VorticityEquation {
    grid: SpectralGrid,
    nu: f64,
}

impl VorticityEquation {
    pub fn new(n: usize, nu: f64) -> Self {
        assert!(nu >= 0.0, "viscosity must be non-negative");
        Self {
            grid: SpectralGrid::new(n),
            nu,
        }
    }

    pub fn grid(&self) -> &SpectralGrid {
        &self.grid
    }

    pub fn initial_vorticity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        self.grid.vorticity_from_velocity(u, v)
    }

    pub fn project(&self, omega: &mut [f64]) {
        let mut modes = self.grid.forward_real(omega);
        self.grid.filter_modes(&mut modes);
        omega.copy_from_slice(&self.grid.inverse_real(&modes));
    }

    pub fn fields(&self, omega: &[f64]) -> FlowFields {
        let mut omega_modes = self.grid.forward_real(omega);
        self.grid.filter_modes(&mut omega_modes);
        let filtered_omega = self.grid.inverse_real(&omega_modes);
        let (u_modes, v_modes) = self.grid.velocity_modes(&omega_modes);
        FlowFields {
            u: self.grid.inverse_real(&u_modes),
            v: self.grid.inverse_real(&v_modes),
            omega: filtered_omega,
        }
    }

    /// Evaluate `-u dot grad(omega) + nu laplacian(omega)`.
    pub fn rate(&self, omega: &[f64], output: &mut [f64]) {
        let expected_length = self.grid.n * self.grid.n;
        assert_eq!(omega.len(), expected_length);
        assert_eq!(output.len(), expected_length);

        let mut omega_modes = self.grid.forward_real(omega);
        self.grid.filter_modes(&mut omega_modes);
        let (u_modes, v_modes) = self.grid.velocity_modes(&omega_modes);
        let mut omega_x_modes = omega_modes.clone();
        let mut omega_y_modes = omega_modes.clone();
        for y in 0..self.grid.n {
            let ky = self.grid.wave_number(y) as f64;
            for x in 0..self.grid.n {
                let kx = self.grid.wave_number(x) as f64;
                let index = y * self.grid.n + x;
                omega_x_modes[index] *= Complex::new(0.0, kx);
                omega_y_modes[index] *= Complex::new(0.0, ky);
            }
        }

        let u = self.grid.inverse_real(&u_modes);
        let v = self.grid.inverse_real(&v_modes);
        let omega_x = self.grid.inverse_real(&omega_x_modes);
        let omega_y = self.grid.inverse_real(&omega_y_modes);
        let advection: Vec<f64> = (0..expected_length)
            .map(|index| u[index] * omega_x[index] + v[index] * omega_y[index])
            .collect();
        let mut advection_modes = self.grid.forward_real(&advection);
        self.grid.filter_modes(&mut advection_modes);

        let mut rate_modes = vec![Complex::new(0.0, 0.0); expected_length];
        for y in 0..self.grid.n {
            let ky = self.grid.wave_number(y) as f64;
            for x in 0..self.grid.n {
                let kx = self.grid.wave_number(x) as f64;
                let index = y * self.grid.n + x;
                rate_modes[index] =
                    -advection_modes[index] - self.nu * (kx * kx + ky * ky) * omega_modes[index];
            }
        }
        output.copy_from_slice(&self.grid.inverse_real(&rate_modes));
    }
}

pub fn diagnostics(fields: &FlowFields) -> (f64, f64) {
    let count = fields.omega.len() as f64;
    let energy = 0.5
        * fields
            .u
            .iter()
            .zip(&fields.v)
            .map(|(u, v)| u * u + v * v)
            .sum::<f64>()
        / count;
    let enstrophy = 0.5 * fields.omega.iter().map(|omega| omega * omega).sum::<f64>() / count;
    (energy, enstrophy)
}

pub fn taylor_green_field(n: usize, nu: f64, time: f64) -> InputField {
    assert!(n > 0 && n.is_multiple_of(2));
    assert!(nu >= 0.0 && time >= 0.0);
    let amplitude = (-2.0 * nu * time).exp();
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for y_index in 0..n {
        let y = TAU * y_index as f64 / n as f64;
        for x_index in 0..n {
            let x = TAU * x_index as f64 / n as f64;
            u.push(x.cos() * y.sin() * amplitude);
            v.push(-x.sin() * y.cos() * amplitude);
        }
    }
    InputField {
        case: "taylor-green".to_owned(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    }
}

pub fn random_velocity_field(
    n: usize,
    seed: u64,
    k_min: usize,
    k_max: usize,
) -> Result<InputField, String> {
    if n == 0 || !n.is_multiple_of(2) {
        return Err("n must be positive and even".to_owned());
    }
    if k_min == 0 || k_min > k_max {
        return Err("require 1 <= k-min <= k-max".to_owned());
    }
    let grid = SpectralGrid::new(n);
    if k_max > grid.cutoff() {
        return Err(format!(
            "k-max must not exceed the two-thirds cutoff {}",
            grid.cutoff()
        ));
    }

    let mut modes = vec![Complex::new(0.0, 0.0); n * n];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mode_count = 0;
    for ky in -(k_max as isize)..=k_max as isize {
        for kx in -(k_max as isize)..=k_max as isize {
            let squared_radius = (kx * kx + ky * ky) as usize;
            if squared_radius < k_min * k_min || squared_radius > k_max * k_max {
                continue;
            }
            if ky < 0 || (ky == 0 && kx <= 0) {
                continue;
            }
            let phase = rng.gen_range(0.0..TAU);
            let value = Complex::from_polar(1.0, phase);
            let positive = grid.mode_index(ky) * n + grid.mode_index(kx);
            let negative = grid.mode_index(-ky) * n + grid.mode_index(-kx);
            modes[positive] = value;
            modes[negative] = value.conj();
            mode_count += 1;
        }
    }
    if mode_count == 0 {
        return Err("the requested k band contains no Fourier modes".to_owned());
    }

    let (mut u, mut v) = grid.velocity_from_vorticity_modes(&modes);
    let temporary = FlowFields {
        u: u.clone(),
        v: v.clone(),
        omega: grid.inverse_real(&modes),
    };
    let (energy, _) = diagnostics(&temporary);
    let scale = (0.5 / energy).sqrt();
    for value in u.iter_mut().chain(v.iter_mut()) {
        *value *= scale;
    }

    Ok(InputField {
        case: "random".to_owned(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Integrator, RungeKutta4};

    fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn represented_wave_derivatives_are_exact() {
        let n = 32;
        let grid = SpectralGrid::new(n);
        let mut values = Vec::with_capacity(n * n);
        let mut dx = Vec::with_capacity(n * n);
        let mut dxx = Vec::with_capacity(n * n);
        let mut dxdy = Vec::with_capacity(n * n);
        let mut laplacian = Vec::with_capacity(n * n);
        for y_index in 0..n {
            let y = TAU * y_index as f64 / n as f64;
            for x_index in 0..n {
                let x = TAU * x_index as f64 / n as f64;
                let value = (3.0 * x).sin() * (2.0 * y).cos();
                values.push(value);
                dx.push(3.0 * (3.0 * x).cos() * (2.0 * y).cos());
                dxx.push(-9.0 * value);
                dxdy.push(-6.0 * (3.0 * x).cos() * (2.0 * y).sin());
                laplacian.push(-13.0 * value);
            }
        }
        assert!(max_error(&grid.derivative(&values, 1, 0), &dx) < 1.0e-10);
        assert!(max_error(&grid.derivative(&values, 2, 0), &dxx) < 1.0e-10);
        assert!(max_error(&grid.derivative(&values, 1, 1), &dxdy) < 1.0e-10);
        assert!(max_error(&grid.laplacian(&values), &laplacian) < 1.0e-10);
    }

    #[test]
    fn random_field_has_requested_energy() {
        let field = random_velocity_field(32, 2026, 2, 6).unwrap();
        let equation = VorticityEquation::new(field.n, 0.0);
        let omega = equation.initial_vorticity(&field.u, &field.v);
        let fields = equation.fields(&omega);
        let (energy, _) = diagnostics(&fields);
        assert!((energy - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn random_mode_phases_do_not_depend_on_grid_size() {
        let small = random_velocity_field(32, 2026, 2, 6).unwrap();
        let large = random_velocity_field(64, 2026, 2, 6).unwrap();
        let small_grid = SpectralGrid::new(small.n);
        let large_grid = SpectralGrid::new(large.n);
        let small_omega = small_grid.vorticity_from_velocity(&small.u, &small.v);
        let large_omega = large_grid.vorticity_from_velocity(&large.u, &large.v);
        let small_modes = small_grid.forward_real(&small_omega);
        let large_modes = large_grid.forward_real(&large_omega);

        let small_mode = small_modes[small_grid.mode_index(2)];
        let large_mode = large_modes[large_grid.mode_index(2)];
        let small_phase = small_mode / small_mode.norm();
        let large_phase = large_mode / large_mode.norm();
        assert!((small_phase - large_phase).norm() < 1.0e-12);
    }

    #[test]
    fn two_thirds_projection_removes_short_modes() {
        let n = 18;
        let equation = VorticityEquation::new(n, 0.0);
        let mut omega = Vec::with_capacity(n * n);
        let mut expected = Vec::with_capacity(n * n);
        for y_index in 0..n {
            let y = TAU * y_index as f64 / n as f64;
            for x_index in 0..n {
                let x = TAU * x_index as f64 / n as f64;
                let retained = (6.0 * x).cos() * (2.0 * y).sin();
                expected.push(retained);
                omega.push(retained + (7.0 * x).cos() * (2.0 * y).sin());
            }
        }
        equation.project(&mut omega);
        assert!(max_error(&omega, &expected) < 1.0e-12);
    }

    #[test]
    fn taylor_green_decays_at_the_exact_rate() {
        let n = 16;
        let nu = 0.1;
        let initial = taylor_green_field(n, nu, 0.0);
        let equation = VorticityEquation::new(n, nu);
        let mut omega = equation.initial_vorticity(&initial.u, &initial.v);
        for _ in 0..100 {
            RungeKutta4.step(&mut omega, 0.01, &|state, rate| equation.rate(state, rate));
            equation.project(&mut omega);
        }
        let fields = equation.fields(&omega);
        let (energy, enstrophy) = diagnostics(&fields);
        assert!((energy - 0.25 * (-0.4_f64).exp()).abs() < 1.0e-8);
        assert!((enstrophy - 0.5 * (-0.4_f64).exp()).abs() < 2.0e-8);

        let exact = taylor_green_field(n, nu, 1.0);
        let velocity_error = fields
            .u
            .iter()
            .zip(&fields.v)
            .zip(exact.u.iter().zip(&exact.v))
            .map(|((u, v), (exact_u, exact_v))| (u - exact_u).powi(2) + (v - exact_v).powi(2))
            .sum::<f64>()
            .sqrt();
        let exact_norm = exact
            .u
            .iter()
            .zip(&exact.v)
            .map(|(u, v)| u * u + v * v)
            .sum::<f64>()
            .sqrt();
        assert!(velocity_error / exact_norm < 1.0e-5);
    }
}
