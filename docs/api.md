
# API Reference

This page documents the public classes and functions
of the Vlasov-Poisson package.

## Grid configuration

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   vlasov.config.Grid
```

## Initial conditions

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   vlasov.initial_conditions.maxwellian
   vlasov.initial_conditions.landau_damping
   vlasov.initial_conditions.bump_on_tail
```

## Numerical methods

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   vlasov.numerics.compute_rho
   vlasov.numerics.compute_efield
   vlasov.numerics.linear_interpolate
   vlasov.numerics.cubic_interpolate
   vlasov.numerics.advection_x
   vlasov.numerics.advection_v
```

## Diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   vlasov.diagnostics.electric_energy
   vlasov.diagnostics.total_rho
   vlasov.diagnostics.total_energy
```

## Solver

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   vlasov.solver.step
   vlasov.solver.run
```
