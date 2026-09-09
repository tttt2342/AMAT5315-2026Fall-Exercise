# Week 2

## Lennard-Jones field plot

From the `week2/` directory, generate the pair-energy and force-field plot with:

```bash
cargo run --manifest-path md/Cargo.toml --example field
```

The example calls the `energy` and `force` functions from the `md` crate and saves
the image as `week2/field.png`.

The arrow direction uses the sign of the scalar force and the radial unit vector.
Arrow lengths are compressed for display, with a small $10^{-8}$ display cutoff
to avoid turning floating-point noise at $r_0$ into a visible arrow.
