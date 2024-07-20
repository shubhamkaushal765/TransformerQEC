## Surface Code Simulation `surface_code.py`

### class `MinimalSurfaceCode`

> A Python class for simulating and visualizing simplified surface codes in quantum error correction.

#### Features

- Generate surface codes with custom distances
- Simulate X errors (translatable to Z errors with rot90)
- Calculate syndrome measurements
- Visualize the surface code layout

#### Quick Start

```python
from minimal_surface_code import MinimalSurfaceCode

code = MinimalSurfaceCode(distance=5)
code.plot_surface_code()
```

#### Visualization Key

- Black circles: Data qubits
- Green dots: Syndrome qubits
- Red circles: Errors on data qubits
- Red dots: Non-zero syndrome measurements

Customize error generation by modifying `min_q_err`, `prob_err` parameters in __init__ method.

---

### class `SurfaceCode`

Implements a simulation of a surface code, a type of quantum error correction code. It generates datasets of surface code syndromes and errors for various parameters.

#### Features

- Simulate surface code with customizable distance and rounds
- Generate X and Z errors on qubits
- Collect syndrome measurements and error data
- Parallel processing using Ray for improved performance
- Save results to CSV files for further analysis

#### Usage

1. Set the parameters:
   - `DISTANCE`: The distance of the surface code
   - `ROUNDS`: The number of error propagation rounds
   - `ITERS`: The number of iterations for data generation
   - `INTVL`: The interval for progress updates
   - `TARGET`: The directory where output files will be saved

1. Run the script:
   ```
   python surface_code.py
   ```

2. The script will generate CSV files in the specified target directory, containing the simulation results.

#### Output Format

The generated CSV files contain the following columns:
- Distance: The distance of the surface code
- Rounds: The number of error propagation rounds
- dataSyndrome: Flattened string representation of data syndromes
- phaseSyndrome: Flattened string representation of phase syndromes
- dataError: Flattened string representation of data errors
- phaseError: Flattened string representation of phase errors

## Contributing

Contributions to improve the simulation or extend its capabilities are welcome. Please feel free to submit issues or pull requests.

## Contact

Shubham Kaushal

[Github](https://github.com/shubhamkaushal765) | [LinkedIn](https://www.linkedin.com/in/kaushalshubham/)