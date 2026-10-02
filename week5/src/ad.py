"""Hand-written nodewise JAX AD; derivatives are of energy, not force."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

def energy(r):
    a = r ** -6
    b = a ** 2
    c = b - a
    return 4 * c

def passes(r):
    r = jnp.asarray(r, dtype=jnp.float64)
    a, da = jax.jvp(lambda x: x**-6, (r,), (jnp.ones_like(r),))
    b, db = jax.jvp(lambda x: x**2, (a,), (da,))
    c, dc = jax.jvp(lambda x,y: x-y, (b,a), (db,da))
    u, du = jax.jvp(lambda x: 4*x, (c,), (dc,))
    _, pu = jax.vjp(lambda x: 4*x, c)
    bc, = pu(jnp.ones_like(u))
    _, pc = jax.vjp(lambda x,y: x-y, b,a)
    bb, ba = pc(bc)
    _, pb = jax.vjp(lambda x: x**2, a)
    ba = ba + pb(bb)[0]
    _, pa = jax.vjp(lambda x: x**-6, r)
    br, = pa(ba)
    return u, dict(r=jnp.ones_like(r), a=da,b=db,c=dc,U=du), dict(r=br,a=ba,b=bb,c=bc,U=jnp.ones_like(u))
