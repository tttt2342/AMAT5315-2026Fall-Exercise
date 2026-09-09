/// Return the greeting printed by the `md` executable.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

pub type Vec2 = [f64; 2];

/// The positions, velocities, and cached accelerations of a particle system.
#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    accelerations: Vec<Vec2>,
}

impl System {
    /// Build a system and compute the initial accelerations from its positions.
    pub fn new(positions: Vec<Vec2>, velocities: Vec<Vec2>) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have the same length"
        );

        let accelerations = compute_accelerations(&positions);
        Self {
            positions,
            velocities,
            accelerations,
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    /// Return the cached accelerations used by the next integration step.
    pub fn accelerations(&self) -> &[Vec2] {
        &self.accelerations
    }

    /// Recompute accelerations from the current positions.
    pub fn recompute_accelerations(&mut self) {
        self.accelerations = compute_accelerations(&self.positions);
    }
}

/// Compute all pair forces as accelerations for an open-boundary system.
pub fn compute_accelerations(positions: &[Vec2]) -> Vec<Vec2> {
    let mut accelerations = vec![[0.0; 2]; positions.len()];

    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let displacement = [
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1],
            ];
            let distance = (displacement[0].powi(2) + displacement[1].powi(2)).sqrt();
            assert!(distance > 0.0, "atoms must not overlap");

            let scale = force(distance) / distance;
            let pair_force = [scale * displacement[0], scale * displacement[1]];
            for component in 0..2 {
                accelerations[i][component] += pair_force[component];
                accelerations[j][component] -= pair_force[component];
            }
        }
    }

    accelerations
}

/// Kinetic energy in reduced units.
pub fn kinetic_energy(system: &System) -> f64 {
    system
        .velocities
        .iter()
        .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
        .sum()
}

/// Lennard-Jones potential energy summed over every unique particle pair.
pub fn potential_energy(system: &System) -> f64 {
    let mut potential = 0.0;
    for i in 0..system.n_atoms() {
        for j in (i + 1)..system.n_atoms() {
            let dx = system.positions[i][0] - system.positions[j][0];
            let dy = system.positions[i][1] - system.positions[j][1];
            let distance = (dx.powi(2) + dy.powi(2)).sqrt();
            assert!(distance > 0.0, "atoms must not overlap");
            potential += energy(distance);
        }
    }
    potential
}

/// Total kinetic plus potential energy in reduced units.
pub fn total_energy(system: &System) -> f64 {
    kinetic_energy(system) + potential_energy(system)
}

#[derive(Clone, Copy, Debug)]
pub struct EnergySample {
    pub time: f64,
    pub relative_error: f64,
}

/// A numerical integration rule that advances a system by one time step.
pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

/// Shared driver from Part 1: it accepts any implementation of Integrator.
pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

/// Forward Euler integration.
pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        for i in 0..system.n_atoms() {
            let velocity = system.velocities[i];
            let acceleration = system.accelerations[i];
            for component in 0..2 {
                system.positions[i][component] += dt * velocity[component];
                system.velocities[i][component] += dt * acceleration[component];
            }
        }
        system.recompute_accelerations();
    }
}

/// Velocity-Verlet integration with one new force evaluation per step.
pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        for i in 0..system.n_atoms() {
            let acceleration = system.accelerations[i];
            for component in 0..2 {
                system.velocities[i][component] += 0.5 * dt * acceleration[component];
                system.positions[i][component] += dt * system.velocities[i][component];
            }
        }

        system.recompute_accelerations();

        for i in 0..system.n_atoms() {
            let acceleration = system.accelerations[i];
            for component in 0..2 {
                system.velocities[i][component] += 0.5 * dt * acceleration[component];
            }
        }
    }
}

/// Run one experiment and return the signed relative total-energy error over time.
pub fn run_experiment(
    method: &impl Integrator,
    initial: &System,
    dt: f64,
    steps: usize,
) -> Vec<EnergySample> {
    let mut system = initial.clone();
    system.recompute_accelerations();
    let initial_energy = total_energy(&system);
    assert!(
        initial_energy.abs() > 0.0,
        "relative energy error needs a non-zero initial energy"
    );

    let mut samples = Vec::with_capacity(steps + 1);
    samples.push(EnergySample {
        time: 0.0,
        relative_error: 0.0,
    });

    for step in 1..=steps {
        advance(method, &mut system, dt);
        let relative_error = (total_energy(&system) - initial_energy) / initial_energy.abs();
        samples.push(EnergySample {
            time: step as f64 * dt,
            relative_error,
        });
    }

    samples
}

/// The isolated two-atom experiment specified in Part 3.
pub fn dimer() -> System {
    System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0, 0.0], [0.0, 0.0]])
}

/// Lennard-Jones pair energy in reduced units.
pub fn energy(r: f64) -> f64 {
    let inverse_r6 = (1.0 / r).powi(6);
    4.0 * (inverse_r6 * inverse_r6 - inverse_r6)
}

/// Scalar Lennard-Jones pair force in reduced units.
pub fn force(r: f64) -> f64 {
    let inverse_r = 1.0 / r;
    let inverse_r6 = inverse_r.powi(6);
    24.0 * inverse_r * (2.0 * inverse_r6 * inverse_r6 - inverse_r6)
}

#[cfg(test)]
mod tests {
    use super::{energy, force, greeting};

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn force_matches_energy_derivative() {
        let h = 1e-5;
        let r0 = 2.0_f64.powf(1.0 / 6.0);

        for r in [0.9, 1.0, r0, 1.3, 2.0] {
            let numerical_force = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let direct_force = force(r);
            let tolerance = 1e-6 * f64::max(1.0, direct_force.abs());

            assert!(
                (direct_force - numerical_force).abs() < tolerance,
                "force mismatch at r={r}: direct={direct_force}, numerical={numerical_force}"
            );
        }
    }
}

#[cfg(test)]
mod dimer_tests {
    use super::{Euler, VelocityVerlet, dimer, run_experiment};

    #[test]
    fn dimer_integrators_have_expected_energy_behavior() {
        let initial = dimer();
        let euler_trace = run_experiment(&Euler, &initial, 0.01, 500);
        let verlet_trace = run_experiment(&VelocityVerlet, &initial, 0.01, 500);

        let verlet_max_error = verlet_trace
            .iter()
            .map(|sample| sample.relative_error.abs())
            .fold(0.0, f64::max);
        let euler_final_error = euler_trace
            .last()
            .expect("the trace contains the initial sample")
            .relative_error;

        assert!(
            verlet_max_error < 1e-3,
            "velocity-Verlet error was {verlet_max_error}"
        );
        assert!(
            euler_final_error > 0.5,
            "forward Euler final error was {euler_final_error}"
        );
    }
}
