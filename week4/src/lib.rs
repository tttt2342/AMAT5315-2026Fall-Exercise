//! Time integrators and one-dimensional periodic advection-diffusion rates.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::TAU;
use std::sync::Arc;

/// An autonomous rate function `du/dt = F(u)`.
///
/// The function must overwrite every entry of its output slice.
pub type RateFunction<'a> = dyn Fn(&[f64], &mut [f64]) + 'a;

/// Advance a state vector by one time step.
pub trait Integrator {
    fn step(&self, state: &mut [f64], step_size: f64, rate: &RateFunction<'_>);
}

/// Forward Euler, `u_(n+1) = u_n + h F(u_n)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn step(&self, state: &mut [f64], step_size: f64, rate: &RateFunction<'_>) {
        let mut k1 = vec![0.0; state.len()];
        rate(state, &mut k1);

        for (value, slope) in state.iter_mut().zip(k1) {
            *value += step_size * slope;
        }
    }
}

/// Explicit midpoint rule (second-order Runge-Kutta).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExplicitMidpoint;

impl Integrator for ExplicitMidpoint {
    fn step(&self, state: &mut [f64], step_size: f64, rate: &RateFunction<'_>) {
        let mut k1 = vec![0.0; state.len()];
        rate(state, &mut k1);

        let midpoint: Vec<f64> = state
            .iter()
            .zip(&k1)
            .map(|(value, slope)| value + 0.5 * step_size * slope)
            .collect();
        let mut k2 = vec![0.0; state.len()];
        rate(&midpoint, &mut k2);

        for (value, slope) in state.iter_mut().zip(k2) {
            *value += step_size * slope;
        }
    }
}

/// Classical fourth-order Runge-Kutta method.
#[derive(Clone, Copy, Debug, Default)]
pub struct RungeKutta4;

impl Integrator for RungeKutta4 {
    fn step(&self, state: &mut [f64], step_size: f64, rate: &RateFunction<'_>) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        rate(state, &mut k1);

        let mut trial: Vec<f64> = state
            .iter()
            .zip(&k1)
            .map(|(value, slope)| value + 0.5 * step_size * slope)
            .collect();
        let mut k2 = vec![0.0; n];
        rate(&trial, &mut k2);

        for ((trial_value, value), slope) in trial.iter_mut().zip(state.iter()).zip(&k2) {
            *trial_value = value + 0.5 * step_size * slope;
        }
        let mut k3 = vec![0.0; n];
        rate(&trial, &mut k3);

        for ((trial_value, value), slope) in trial.iter_mut().zip(state.iter()).zip(&k3) {
            *trial_value = value + step_size * slope;
        }
        let mut k4 = vec![0.0; n];
        rate(&trial, &mut k4);

        for i in 0..n {
            state[i] += step_size * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
    }
}

/// Fourier rate for `u_t + c u_x = nu u_xx` on `[0, 2 pi)`.
///
/// The grid size must be positive and even. FFT coefficients use the order
/// `0, 1, ..., n/2 - 1, -n/2, ..., -1`. The first-derivative multiplier of
/// the Nyquist mode `k = -n/2` is set to zero, while its second-derivative
/// multiplier remains `-k^2`.
pub struct FourierAdvectionDiffusion {
    n: usize,
    c: f64,
    nu: f64,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl FourierAdvectionDiffusion {
    pub fn new(n: usize, c: f64, nu: f64) -> Self {
        assert!(
            n > 0 && n.is_multiple_of(2),
            "Fourier grid size must be positive and even"
        );
        assert!(nu >= 0.0, "viscosity must be non-negative");

        let mut planner = FftPlanner::new();
        Self {
            n,
            c,
            nu,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Evaluate the rate, overwriting `output`.
    pub fn rate(&self, state: &[f64], output: &mut [f64]) {
        assert_eq!(state.len(), self.n, "state length must match the grid");
        assert_eq!(output.len(), self.n, "output length must match the grid");

        let mut modes: Vec<Complex<f64>> = state
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        self.forward.process(&mut modes);

        for (index, mode) in modes.iter_mut().enumerate() {
            let k = if index <= self.n / 2 {
                index as f64
            } else {
                index as f64 - self.n as f64
            };
            let advection_frequency = if index == self.n / 2 {
                0.0
            } else {
                -self.c * k
            };
            let multiplier = Complex::new(-self.nu * k * k, advection_frequency);
            *mode *= multiplier;
        }

        self.inverse.process(&mut modes);
        let normalization = 1.0 / self.n as f64;
        for (value, mode) in output.iter_mut().zip(modes) {
            *value = normalization * mode.re;
        }
    }
}

/// Centred finite-difference rate for `u_t + c u_x = nu u_xx` on
/// `[0, 2 pi)`, with periodic neighbours.
#[derive(Clone, Copy, Debug)]
pub struct CenteredAdvectionDiffusion {
    n: usize,
    c: f64,
    nu: f64,
    dx: f64,
}

impl CenteredAdvectionDiffusion {
    pub fn new(n: usize, c: f64, nu: f64) -> Self {
        assert!(n >= 2, "centred-difference grid needs at least two points");
        assert!(nu >= 0.0, "viscosity must be non-negative");
        Self {
            n,
            c,
            nu,
            dx: TAU / n as f64,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Evaluate the rate, overwriting `output`.
    pub fn rate(&self, state: &[f64], output: &mut [f64]) {
        assert_eq!(state.len(), self.n, "state length must match the grid");
        assert_eq!(output.len(), self.n, "output length must match the grid");

        let first_denominator = 2.0 * self.dx;
        let second_denominator = self.dx * self.dx;
        for j in 0..self.n {
            let left = state[(j + self.n - 1) % self.n];
            let centre = state[j];
            let right = state[(j + 1) % self.n];
            let first = (right - left) / first_denominator;
            let second = (right - 2.0 * centre + left) / second_denominator;
            output[j] = -self.c * first + self.nu * second;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn grid_wave(n: usize, wave_number: usize) -> Vec<f64> {
        (0..n)
            .map(|j| (wave_number as f64 * TAU * j as f64 / n as f64).cos())
            .collect()
    }

    fn exact_wave(n: usize, wave_number: usize, c: f64, nu: f64, time: f64) -> Vec<f64> {
        let k = wave_number as f64;
        let amplitude = (-nu * k * k * time).exp();
        (0..n)
            .map(|j| amplitude * (k * (TAU * j as f64 / n as f64 - c * time)).cos())
            .collect()
    }

    fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0, f64::max)
    }

    fn integrate_wave(integrator: &dyn Integrator, dt: f64) -> f64 {
        let n = 32;
        let wave_number = 2;
        let c = 0.4;
        let nu = 0.05;
        let final_time = 1.0;
        let steps = (final_time / dt).round() as usize;
        assert!((steps as f64 * dt - final_time).abs() < 1.0e-14);

        let equation = FourierAdvectionDiffusion::new(n, c, nu);
        let mut state = grid_wave(n, wave_number);
        for _ in 0..steps {
            integrator.step(&mut state, dt, &|u, du| equation.rate(u, du));
        }

        max_error(&state, &exact_wave(n, wave_number, c, nu, final_time))
    }

    #[test]
    fn forward_euler_matches_a_single_wave() {
        assert!(integrate_wave(&ForwardEuler, 0.001) < 4.0e-4);
    }

    #[test]
    fn explicit_midpoint_matches_a_single_wave() {
        assert!(integrate_wave(&ExplicitMidpoint, 0.01) < 2.0e-5);
    }

    #[test]
    fn rk4_matches_a_single_wave() {
        assert!(integrate_wave(&RungeKutta4, 0.05) < 1.0e-6);
    }

    #[test]
    fn fourier_rate_is_exact_for_a_resolved_wave() {
        let n = 32;
        let k = 3.0;
        let c = 0.7;
        let nu = 0.04;
        let equation = FourierAdvectionDiffusion::new(n, c, nu);
        let state = grid_wave(n, k as usize);
        let mut rate = vec![0.0; n];
        equation.rate(&state, &mut rate);

        let expected: Vec<f64> = (0..n)
            .map(|j| {
                let phase = k * TAU * j as f64 / n as f64;
                c * k * phase.sin() - nu * k * k * phase.cos()
            })
            .collect();
        assert!(max_error(&rate, &expected) < 2.0e-13);
    }

    #[test]
    fn fourier_nyquist_mode_diffuses_but_does_not_travel() {
        let n = 16;
        let nu = 0.03;
        let equation = FourierAdvectionDiffusion::new(n, 2.5, nu);
        let state: Vec<f64> = (0..n)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut rate = vec![0.0; n];
        equation.rate(&state, &mut rate);

        let expected_factor = -nu * (n as f64 / 2.0).powi(2);
        let expected: Vec<f64> = state.iter().map(|value| expected_factor * value).collect();
        assert!(max_error(&rate, &expected) < 2.0e-13);
    }

    #[test]
    fn centred_rate_wraps_periodic_neighbours() {
        let n = 32;
        let k = 3.0;
        let c = 0.7;
        let nu = 0.04;
        let dx = TAU / n as f64;
        let equation = CenteredAdvectionDiffusion::new(n, c, nu);
        let state = grid_wave(n, k as usize);
        let mut rate = vec![0.0; n];
        equation.rate(&state, &mut rate);

        let modified_first = (k * dx).sin() / dx;
        let modified_second = -4.0 * (0.5 * k * dx).sin().powi(2) / (dx * dx);
        let expected: Vec<f64> = (0..n)
            .map(|j| {
                let phase = k * dx * j as f64;
                c * modified_first * phase.sin() + nu * modified_second * phase.cos()
            })
            .collect();
        assert!(max_error(&rate, &expected) < 2.0e-13);
    }
}
