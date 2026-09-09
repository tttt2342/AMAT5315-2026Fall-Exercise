/// Return the greeting printed by the `md` executable.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy in reduced units.
pub fn energy(r: f64) -> f64 {
    let inverse_r6 = (1.0 / r).powi(6);
    4.0 * (inverse_r6 * inverse_r6 - inverse_r6)
}

/// Scalar Lennard-Jones pair force in reduced units.
pub fn force(_r: f64) -> f64 {
    todo!("Lennard-Jones force is not implemented yet")
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
