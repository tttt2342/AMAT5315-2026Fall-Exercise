use continuum::{
    CenteredAdvectionDiffusion, ExplicitMidpoint, ForwardEuler, FourierAdvectionDiffusion,
    Integrator, RateFunction, RungeKutta4,
};
use std::env;
use std::f64::consts::{FRAC_PI_2, TAU};

const N: usize = 64;
const C: f64 = 1.0;

struct EqualWeightRk4;

impl Integrator for EqualWeightRk4 {
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
            state[i] += step_size * (k1[i] + k2[i] + k3[i] + k4[i]) / 4.0;
        }
    }
}

fn periodic_gaussian(x: f64, centre: f64, variance: f64, amplitude: f64) -> f64 {
    (-4..=4)
        .map(|image| {
            let distance = x - centre + image as f64 * TAU;
            amplitude * (-distance * distance / (2.0 * variance)).exp()
        })
        .sum()
}

fn evolve(
    integrator: &dyn Integrator,
    initial: &[f64],
    dt: f64,
    final_time: f64,
    rate: &RateFunction<'_>,
) -> Vec<f64> {
    let mut state = initial.to_vec();
    let mut time = 0.0;
    while time < final_time {
        let step_size = dt.min(final_time - time);
        integrator.step(&mut state, step_size, rate);
        time += step_size;
        if final_time - time < 32.0 * f64::EPSILON * final_time {
            time = final_time;
        }
    }
    state
}

fn exact_solution(x: &[f64], sigma: f64, nu: f64, time: f64) -> Vec<f64> {
    let variance = sigma * sigma + 2.0 * nu * time;
    let amplitude = sigma / variance.sqrt();
    x.iter()
        .map(|&point| periodic_gaussian(point, FRAC_PI_2 + C * time, variance, amplitude))
        .collect()
}

fn max_error(actual: &[f64], exact: &[f64]) -> f64 {
    actual
        .iter()
        .zip(exact)
        .map(|(value, reference)| (value - reference).abs())
        .fold(0.0, f64::max)
}

fn profile_comparison() {
    let nu = 0.002;
    let sigma = 0.25;
    let final_time = TAU;
    let x: Vec<f64> = (0..N).map(|j| TAU * j as f64 / N as f64).collect();
    let initial: Vec<f64> = x
        .iter()
        .map(|&point| periodic_gaussian(point, FRAC_PI_2, sigma * sigma, 1.0))
        .collect();
    let exact = exact_solution(&x, sigma, nu, final_time);

    let fourier = FourierAdvectionDiffusion::new(N, C, nu);
    let centred = CenteredAdvectionDiffusion::new(N, C, nu);
    let rk4_fourier = evolve(&RungeKutta4, &initial, 0.02, final_time, &|u, du| {
        fourier.rate(u, du)
    });
    let rk4_centred = evolve(&RungeKutta4, &initial, 0.02, final_time, &|u, du| {
        centred.rate(u, du)
    });
    let euler_fourier = evolve(&ForwardEuler, &initial, 0.005, final_time, &|u, du| {
        fourier.rate(u, du)
    });

    println!("# x exact rk4_fourier rk4_centred euler_fourier");
    for j in 0..N {
        println!(
            "{:.15e} {:.15e} {:.15e} {:.15e} {:.15e}",
            x[j], exact[j], rk4_fourier[j], rk4_centred[j], euler_fourier[j]
        );
    }
}

fn convergence_study() {
    let nu = 0.05;
    let sigma = 0.35;
    let final_time = 1.0;
    let x: Vec<f64> = (0..N).map(|j| TAU * j as f64 / N as f64).collect();
    let initial: Vec<f64> = x
        .iter()
        .map(|&point| periodic_gaussian(point, FRAC_PI_2, sigma * sigma, 1.0))
        .collect();
    let exact = exact_solution(&x, sigma, nu, final_time);
    let fourier = FourierAdvectionDiffusion::new(N, C, nu);

    println!("# h euler midpoint rk4 equal_weight_rk4");
    for dt in [0.02, 0.01, 0.005, 0.0025] {
        let euler = evolve(&ForwardEuler, &initial, dt, final_time, &|u, du| {
            fourier.rate(u, du)
        });
        let midpoint = evolve(&ExplicitMidpoint, &initial, dt, final_time, &|u, du| {
            fourier.rate(u, du)
        });
        let rk4 = evolve(&RungeKutta4, &initial, dt, final_time, &|u, du| {
            fourier.rate(u, du)
        });
        let equal_weight = evolve(&EqualWeightRk4, &initial, dt, final_time, &|u, du| {
            fourier.rate(u, du)
        });
        println!(
            "{dt:.15e} {:.15e} {:.15e} {:.15e} {:.15e}",
            max_error(&euler, &exact),
            max_error(&midpoint, &exact),
            max_error(&rk4, &exact),
            max_error(&equal_weight, &exact),
        );
    }
}

fn main() {
    match env::args().nth(1).as_deref() {
        None => profile_comparison(),
        Some("convergence") => convergence_study(),
        Some(argument) => panic!("unknown argument: {argument}"),
    }
}
