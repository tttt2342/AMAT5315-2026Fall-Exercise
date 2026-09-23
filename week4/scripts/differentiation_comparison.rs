use continuum::SpectralGrid;
use std::f64::consts::TAU;

#[derive(Clone, Copy)]
enum Operator {
    Dx,
    Dxx,
    DxDy,
    Laplacian,
}

impl Operator {
    fn name(self) -> &'static str {
        match self {
            Self::Dx => "dx",
            Self::Dxx => "dxx",
            Self::DxDy => "dxdy",
            Self::Laplacian => "Laplacian",
        }
    }
}

fn field_and_exact(n: usize, operator: Operator) -> (Vec<f64>, Vec<f64>) {
    let mut field = Vec::with_capacity(n * n);
    let mut exact = Vec::with_capacity(n * n);
    for y_index in 0..n {
        let y = TAU * y_index as f64 / n as f64;
        for x_index in 0..n {
            let x = TAU * x_index as f64 / n as f64;
            let g = (3.0 * x).sin() * (2.0 * y).cos();
            field.push(g);
            exact.push(match operator {
                Operator::Dx => 3.0 * (3.0 * x).cos() * (2.0 * y).cos(),
                Operator::Dxx => -9.0 * g,
                Operator::DxDy => -6.0 * (3.0 * x).cos() * (2.0 * y).sin(),
                Operator::Laplacian => -13.0 * g,
            });
        }
    }
    (field, exact)
}

fn centred_difference(field: &[f64], n: usize, operator: Operator) -> Vec<f64> {
    let dx = TAU / n as f64;
    let mut result = vec![0.0; n * n];
    for y in 0..n {
        let down = (y + n - 1) % n;
        let up = (y + 1) % n;
        for x in 0..n {
            let left = (x + n - 1) % n;
            let right = (x + 1) % n;
            let index = y * n + x;
            result[index] = match operator {
                Operator::Dx => (field[y * n + right] - field[y * n + left]) / (2.0 * dx),
                Operator::Dxx => {
                    (field[y * n + right] - 2.0 * field[index] + field[y * n + left]) / (dx * dx)
                }
                Operator::DxDy => {
                    (field[up * n + right] - field[up * n + left] - field[down * n + right]
                        + field[down * n + left])
                        / (4.0 * dx * dx)
                }
                Operator::Laplacian => {
                    (field[y * n + right]
                        + field[y * n + left]
                        + field[up * n + x]
                        + field[down * n + x]
                        - 4.0 * field[index])
                        / (dx * dx)
                }
            };
        }
    }
    result
}

fn fourier_difference(field: &[f64], n: usize, operator: Operator) -> Vec<f64> {
    let grid = SpectralGrid::new(n);
    match operator {
        Operator::Dx => grid.derivative(field, 1, 0),
        Operator::Dxx => grid.derivative(field, 2, 0),
        Operator::DxDy => grid.derivative(field, 1, 1),
        Operator::Laplacian => grid.laplacian(field),
    }
}

fn max_error(actual: &[f64], exact: &[f64]) -> f64 {
    actual
        .iter()
        .zip(exact)
        .map(|(actual, exact)| (actual - exact).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let operators = [
        Operator::Dx,
        Operator::Dxx,
        Operator::DxDy,
        Operator::Laplacian,
    ];
    println!("Derivative\tFD N=32\tFD N=64\tRatio (32/64)\tFourier N=32");
    for operator in operators {
        let (field_32, exact_32) = field_and_exact(32, operator);
        let (field_64, exact_64) = field_and_exact(64, operator);
        let fd_32 = max_error(&centred_difference(&field_32, 32, operator), &exact_32);
        let fd_64 = max_error(&centred_difference(&field_64, 64, operator), &exact_64);
        let fourier_32 = max_error(&fourier_difference(&field_32, 32, operator), &exact_32);
        let ratio = fd_32 / fd_64;
        assert!(
            (3.8..4.2).contains(&ratio),
            "{} finite-difference ratio is not second order",
            operator.name()
        );
        assert!(
            fourier_32 < 1.0e-10,
            "{} Fourier error exceeds 1e-10",
            operator.name()
        );
        println!(
            "{}\t{fd_32:.8e}\t{fd_64:.8e}\t{ratio:.5}\t{fourier_32:.8e}",
            operator.name()
        );
    }
}
