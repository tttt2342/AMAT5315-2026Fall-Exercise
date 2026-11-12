pub fn site(k: f64) -> ([f64; 16], [f64; 2]) {
    let even = k.cosh().sqrt();
    let odd = k.sinh().sqrt();
    let w = [[even, odd], [even, -odd]];
    let mut a = [0.; 16];
    for (index, value) in a.iter_mut().enumerate() {
        // The four bits label the right, up, left and down indices.
        *value = w
            .iter()
            .map(|spin| (0..4).map(|bit| spin[(index >> bit) & 1]).product::<f64>())
            .sum();
    }
    (a, [even.recip(), 0.])
}
