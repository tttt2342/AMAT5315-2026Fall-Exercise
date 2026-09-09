# Part 4 implementation plan

## 1. Preserve the Part 3 API

- Keep the existing open-boundary System::new, Integrator trait, Euler,
  VelocityVerlet, dimer experiment, and dimer tests.
- Add a periodic fluid constructor/configuration to the same System so the
  existing integrators can advance both models.
- Wrap positions after each drift and use cached accelerations for Verlet.

## 2. Add the fluid physics

- Generate the square triangular lattice from n and rho, including its box.
- Add minimum-image displacement and periodic wrapping.
- Add the shifted Lennard-Jones potential and cutoff force at rc = 2.5.
- Keep the pair loop as the simple all-pairs implementation for Part 4.
- Add seeded Gaussian velocities, centre-of-mass removal, and the equilibration
  velocity rescaling rule.

## 3. Add red tests before the implementation

- Assert that periodic pair forces sum to zero.
- Probe just inside the cutoff and at the cutoff to assert continuity of the
  shifted potential.
- Run the default contract and assert the drift, speed-temperature, and
  Maxwell-Boltzmann shape bounds.
- Invoke the binary with a small valid run and assert that its JSON files are
  readable and contain the required fields and frame count.

Run the tests before adding the missing implementation and commit this red
state. The test names should describe the physics or file contract.

## 4. Implement the CLI

- Add clap subcommands run, check, and video.
- Add serde/serde_json data types for run.json and traj.jsonl.
- Make run use the Part 4 defaults and write only production steps divisible
  by sample_every, excluding step zero.
- Make check validate the files, recompute energies, calculate the three
  acceptance statistics, print PASS/FAIL, and return a nonzero error on failure.

## 5. Implement the trajectory video

- Read only the saved trajectory and run metadata.
- Render a periodic-box atom panel and a cumulative radial-distribution panel
  up to half the shorter box side.
- Write temporary PNG frames outside tracked output and call ffmpeg to encode
  an MP4 under 2 MB.

Run release tests and a small CLI smoke test, then the full default contract.
Commit the green implementation before adding the reproduce rule.

## 6. Reproduce target

- Add week2/Makefile with a reproduce target using
  cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts.
- Ignore week2/artifacts/ and existing build outputs.
- Run make reproduce and md check artifacts.
- Do not add generated trajectory files, video files, or build directories to
  Git.
