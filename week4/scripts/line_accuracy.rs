use continuum::{
    CenteredAdvectionDiffusion, ForwardEuler, FourierAdvectionDiffusion, Integrator, RateFunction,
    RungeKutta4,
};
use std::f64::consts::{FRAC_PI_2, TAU};

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.002;
const SIGMA: f64 = 0.25;
const T_END: f64 = TAU;

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
    rate: &RateFunction<'_>,
) -> Vec<f64> {
    let mut state = initial.to_vec();
    let mut time = 0.0;
    while time < T_END {
        let step_size = dt.min(T_END - time);
        integrator.step(&mut state, step_size, rate);
        time += step_size;
        if T_END - time < 32.0 * f64::EPSILON * T_END {
            time = T_END;
        }
    }
    state
}

fn main() {
    let x: Vec<f64> = (0..N).map(|j| TAU * j as f64 / N as f64).collect();
    let initial: Vec<f64> = x
        .iter()
        .map(|&point| periodic_gaussian(point, FRAC_PI_2, SIGMA * SIGMA, 1.0))
        .collect();

    let final_variance = SIGMA * SIGMA + 2.0 * NU * T_END;
    let final_amplitude = SIGMA / final_variance.sqrt();
    let exact: Vec<f64> = x
        .iter()
        .map(|&point| {
            periodic_gaussian(
                point,
                FRAC_PI_2 + C * T_END,
                final_variance,
                final_amplitude,
            )
        })
        .collect();

    let fourier = FourierAdvectionDiffusion::new(N, C, NU);
    let centred = CenteredAdvectionDiffusion::new(N, C, NU);
    let rk4_fourier = evolve(&RungeKutta4, &initial, 0.02, &|u, du| fourier.rate(u, du));
    let rk4_centred = evolve(&RungeKutta4, &initial, 0.02, &|u, du| centred.rate(u, du));
    let euler_fourier = evolve(&ForwardEuler, &initial, 0.005, &|u, du| fourier.rate(u, du));

    println!("# x exact rk4_fourier rk4_centred euler_fourier");
    for j in 0..N {
        println!(
            "{:.15e} {:.15e} {:.15e} {:.15e} {:.15e}",
            x[j], exact[j], rk4_fourier[j], rk4_centred[j], euler_fourier[j]
        );
    }
}
