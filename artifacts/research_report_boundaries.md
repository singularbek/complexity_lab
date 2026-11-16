# Scientific Investigation Report: Investigate the impact of boundary conditions on density preservation and entropy

## 1. Hypothesis Formulation
**Hypothesis Statement:** Fixed boundaries cause faster population decay and lower entropy than Periodic boundaries because activity dissipates at edges.

**Theoretical Rationale:** In periodic boundaries, structures wrap around indefinitely, while fixed boundaries act as a zero-density Dirichlet sink absorbing traveling patterns.

**Expected Outcome:** Periodic boundaries will preserve higher density than Fixed boundaries.

## 2. Empirical Methodology & Observables
| Experiment Name | ID | Mean Density | Final Entropy | Periodic | Period |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Toroidal_Periodic_Boundary | `exp_215ae7cc` | 0.1866 | 0.7173 | False | 0 |
| Absorbing_Fixed_Boundary | `exp_6f2488c8` | 0.1380 | 0.4754 | False | 0 |

### Experiment Details & Interpretations
- **Toroidal_Periodic_Boundary**: Completed in 0.0766s. Final alive cells: 155, Entropy: 0.7173 bits, Periodic: False.
- **Absorbing_Fixed_Boundary**: Completed in 0.0077s. Final alive cells: 80, Entropy: 0.4754 bits, Periodic: False.

## 3. Theoretical Synthesis
The research agent investigated: 'Investigate the impact of boundary conditions on density preservation and entropy'.
Tested Hypothesis: "Fixed boundaries cause faster population decay and lower entropy than Periodic boundaries because activity dissipates at edges."

Empirical Observations:
- Experiment 'Toroidal_Periodic_Boundary' (ID: exp_215ae7cc): Mean Density = 0.1866, Final Entropy = 0.7173 bits, Periodic Orbit = False (T=0), Mean Activity = 126.95 cells/step.
- Experiment 'Absorbing_Fixed_Boundary' (ID: exp_6f2488c8): Mean Density = 0.1380, Final Entropy = 0.4754 bits, Periodic Orbit = False (T=0), Mean Activity = 75.44 cells/step.

Theoretical Synthesis:
Quantitative contrast between 'Toroidal_Periodic_Boundary' and 'Absorbing_Fixed_Boundary' reveals a density delta of 0.0486 and activity delta of 51.51. The numerical evidence directly validates the differential impact predicted.

## 4. Scientific Conclusion
> **Hypothesis is SUPPORTED by empirical simulation data.**

## 5. Suggested Next Steps
- Perform Monte Carlo sweeps over multiple seeds to establish statistical error bars.
- Test higher grid resolutions to eliminate finite-size boundary effects.
- Perform fine-grained parameter sweep across intermediate transition points.