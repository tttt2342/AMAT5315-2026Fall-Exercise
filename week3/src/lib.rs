use rand::Rng;

/// An L by L Ising lattice stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    side: usize,
    spins: Vec<i8>,
    magnetization: i64,
    energy: i64,
}

impl Lattice {
    /// Construct the all-up initial condition required by the ramp contract.
    pub fn all_up(side: usize) -> Self {
        let sites = side * side;
        Self {
            side,
            spins: vec![1; sites],
            magnetization: sites as i64,
            energy: -2 * sites as i64,
        }
    }

    pub fn side(&self) -> usize {
        self.side
    }

    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    pub fn mean_spin(&self) -> f64 {
        self.magnetization as f64 / self.spins.len() as f64
    }

    pub fn energy_per_site(&self) -> f64 {
        self.energy as f64 / self.spins.len() as f64
    }

    pub fn total_energy(&self) -> i64 {
        self.energy
    }

    fn neighbor_sum(&self, index: usize) -> i64 {
        let row = index / self.side;
        let col = index % self.side;
        let up = ((row + self.side - 1) % self.side) * self.side + col;
        let down = ((row + 1) % self.side) * self.side + col;
        let left = row * self.side + (col + self.side - 1) % self.side;
        let right = row * self.side + (col + 1) % self.side;

        i64::from(self.spins[up])
            + i64::from(self.spins[down])
            + i64::from(self.spins[left])
            + i64::from(self.spins[right])
    }

    fn flip(&mut self, index: usize, delta_energy: i64) {
        let old_spin = self.spins[index];
        self.spins[index] = -old_spin;
        self.magnetization -= 2 * i64::from(old_spin);
        self.energy += delta_energy;
    }

    /// Perform one Metropolis sweep: L^2 independent uniform site proposals.
    /// Returns the number of accepted proposals.
    pub fn metropolis_sweep<R: Rng + ?Sized>(&mut self, temperature: f64, rng: &mut R) -> u64 {
        let mut accepted = 0;
        for _ in 0..self.spins.len() {
            let index = rng.gen_range(0..self.spins.len());
            let delta_energy = 2 * i64::from(self.spins[index]) * self.neighbor_sum(index);
            let accept = delta_energy <= 0
                || rng.gen::<f64>() < (-(delta_energy as f64) / temperature).exp();
            if accept {
                self.flip(index, delta_energy);
                accepted += 1;
            }
        }
        accepted
    }

    #[cfg(test)]
    fn energy_from_scratch(&self) -> i64 {
        let mut energy = 0;
        for row in 0..self.side {
            for col in 0..self.side {
                let here = i64::from(self.spins[row * self.side + col]);
                let right = i64::from(self.spins[row * self.side + (col + 1) % self.side]);
                let down = i64::from(self.spins[((row + 1) % self.side) * self.side + col]);
                energy -= here * (right + down);
            }
        }
        energy
    }
}

/// Build an ascending grid without accumulating floating-point addition error.
pub fn temperature_grid(from: f64, to: f64, step: f64) -> Vec<f64> {
    let scale = from.abs().max(to.abs()).max(step.abs()).max(1.0);
    let tolerance = 32.0 * f64::EPSILON * scale;
    let max_index = ((to - from) / step + tolerance).floor() as usize;
    (0..=max_index)
        .map(|index| from + index as f64 * step)
        .filter(|temperature| *temperature <= to + tolerance)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn all_up_has_expected_observables() {
        let lattice = Lattice::all_up(4);
        assert_eq!(lattice.mean_spin(), 1.0);
        assert_eq!(lattice.energy_per_site(), -2.0);
    }

    #[test]
    fn incremental_energy_matches_direct_sum() {
        let mut lattice = Lattice::all_up(7);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..20 {
            lattice.metropolis_sweep(2.3, &mut rng);
            assert_eq!(lattice.total_energy(), lattice.energy_from_scratch());
        }
    }

    #[test]
    fn temperature_grid_includes_only_reached_upper_bound() {
        assert_eq!(temperature_grid(1.5, 1.6, 0.05), vec![1.5, 1.55, 1.6]);
        assert_eq!(temperature_grid(1.5, 1.61, 0.05), vec![1.5, 1.55, 1.6]);
        assert_eq!(temperature_grid(1.5, 1.59, 0.05), vec![1.5, 1.55]);
    }

    #[test]
    fn seeded_sweeps_are_reproducible() {
        let mut first = Lattice::all_up(8);
        let mut second = Lattice::all_up(8);
        let mut first_rng = ChaCha8Rng::seed_from_u64(42);
        let mut second_rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..10 {
            first.metropolis_sweep(2.5, &mut first_rng);
            second.metropolis_sweep(2.5, &mut second_rng);
        }
        assert_eq!(first, second);
    }
}
