use crate::physics::{Experiment, Kernel};
use serde::Serialize;
use std::collections::BTreeMap;
#[derive(Serialize)]
pub struct Action {
    pub action: &'static str,
    pub step: usize,
    pub saved_states: usize,
}
pub struct Reverse<'a> {
    pub kernel: Kernel<'a>,
    pub c: &'a [f64],
    pub weights: &'a [f64],
    pub image: &'a mut [f64],
    pub adj: Vec<f64>,
    scratch: Vec<f64>,
    primal: Vec<f64>,
    pub calls: usize,
    pub grads: usize,
    pub peak: usize,
    pub actions: Vec<Action>,
    pub frames: Vec<f64>,
    pub frame_steps: Vec<usize>,
    pub every: Option<usize>,
}
impl<'a> Reverse<'a> {
    pub fn new(
        e: &'a Experiment,
        shot: usize,
        c: &'a [f64],
        weights: &'a [f64],
        image: &'a mut [f64],
        every: Option<usize>,
    ) -> Self {
        Self {
            kernel: Kernel::new(e, shot),
            c,
            weights,
            image,
            adj: vec![0.; 2 * e.len()],
            scratch: vec![0.; 2 * e.len()],
            primal: vec![0.; 2 * e.len()],
            calls: 0,
            grads: 0,
            peak: 1,
            actions: vec![],
            frames: vec![],
            frame_steps: vec![],
            every,
        }
    }
    fn grad(&mut self, n: usize, s: &[f64]) {
        let e = self.kernel.e;
        let l = e.len();
        for (k, [x, z]) in e.receivers.iter().enumerate() {
            self.adj[l + z * e.nx + x] += self.weights[n * e.receivers.len() + k];
        }
        self.kernel.vjp(
            n,
            s,
            self.c,
            &mut self.scratch,
            self.image,
            &mut self.primal,
            &mut self.adj,
        );
        std::mem::swap(&mut self.adj, &mut self.scratch);
        self.grads += 1;
        if self.every.is_some_and(|v| n % v == 0) {
            self.frames.extend_from_slice(&self.adj[l..]);
            self.frame_steps.push(n);
        }
    }
    pub fn full(&mut self) {
        let e = self.kernel.e;
        let mut states = Vec::with_capacity(e.steps + 1);
        states.push(vec![0.; 2 * e.len()]);
        for n in 0..e.steps {
            let mut next = vec![0.; 2 * e.len()];
            self.kernel.step(n, &states[n], self.c, &mut next);
            states.push(next);
            self.calls += 1;
        }
        self.peak = states.len();
        for n in (0..e.steps).rev() {
            self.grad(n, &states[n]);
        }
    }
    fn log(&mut self, action: &'static str, step: usize, count: usize) {
        self.peak = self.peak.max(count);
        self.actions.push(Action {
            action,
            step,
            saved_states: count,
        });
    }
    pub fn treeverse(&mut self, budget: usize) {
        assert!(budget > 0);
        let n = self.kernel.e.steps;
        let mut tau = 1;
        while capacity(budget, tau) < n as u128 {
            tau += 1;
        }
        let mut states = BTreeMap::new();
        states.insert(0, vec![0.; 2 * self.kernel.e.len()]);
        self.recurse(&mut states, budget, tau, 0, 0, n);
        assert_eq!(states.len(), 1);
        assert!(self.peak <= budget + 1);
    }
    // Direct port of TreeverseAlgorithm.jl treeverse!, including mid's endpoint rule.
    fn recurse(
        &mut self,
        states: &mut BTreeMap<usize, Vec<f64>>,
        mut delta: usize,
        mut tau: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
    ) {
        if sigma > beta {
            assert!(delta > 0);
            delta -= 1;
            let mut work = states[&beta].clone();
            let mut next = vec![0.; work.len()];
            self.log("restore", beta, states.len());
            for j in beta..sigma {
                self.kernel.step(j, &work, self.c, &mut next);
                std::mem::swap(&mut work, &mut next);
                self.calls += 1;
                self.log("call", j, states.len());
            }
            states.insert(sigma, work);
            self.log("store", sigma, states.len());
        } else {
            assert_eq!(sigma, beta);
        }
        let mut k = mid(delta, tau, sigma, phi);
        while tau > 0 && k < phi {
            self.recurse(states, delta, tau, sigma, k, phi);
            tau -= 1;
            phi = k;
            k = mid(delta, tau, sigma, phi);
        }
        self.grad(sigma, &states[&sigma]);
        self.log("grad", sigma, states.len());
        if sigma > beta {
            states.remove(&sigma);
            self.log("fetch", sigma, states.len());
        }
    }
}
fn capacity(delta: usize, tau: usize) -> u128 {
    let mut v = 1u128;
    for i in 1..=delta {
        v = v.saturating_mul((tau + i) as u128) / (i as u128);
    }
    v
}
fn mid(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    let mut k = if delta + tau == 0 {
        phi
    } else {
        (delta * sigma + tau * phi).div_ceil(delta + tau)
    };
    if k >= phi && delta > 0 {
        k = (sigma + 1).max(phi - 1);
    }
    k
}
