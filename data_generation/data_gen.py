from stim_experiments import get_dets_and_errs, get_circuit
from data_utils import encode_hex, decode_hex
import numpy as np
import polars as pl
from tqdm import tqdm
import os, ray, yaml

with open("config.yaml", 'r') as file:
    data = yaml.safe_load(file)
DISTANCE = data['DISTANCE']
SHOTS = data['SHOTS']
DATASET_DIR = data['DATASET_DIR']

# number of parallel executions is half of the available CPUs
n_parallels = os.cpu_count() // 2
print(f"Number of parallel executions: {n_parallels}")


def get_1_shot(circuit):
    """
    Generate data for a single shot of a quantum error correction circuit.

    Parameters:
    - circuit: The quantum error correction circuit.

    Returns:
    dict: A dictionary containing generated data for the shot, including:
        - "syndrome": List of syndromes in hexadecimal format.
        - "errorX": List of X-basis errors in hexadecimal format.
        - "errorY": List of Y-basis errors in hexadecimal format.
        - "errorZ": List of Z-basis errors in hexadecimal format.
    """
    generated_data = {
            "syndrome":[],
            "errorX": [],
            "errorY": [],
            "errorZ": [],
        }

    # Get data from circuit simulation
    det_data, obs_data, error_table   = get_dets_and_errs(circuit)
    det_data = det_data[0].astype(int).astype(str)
    obs_data = obs_data[0].astype(int)[0]
    det_data = "".join(det_data)

    # Write encoded data to generated_data
    bases = ["X", "Y", "Z"]
    for basis in bases:
        if basis not in error_table.keys():
            generated_data[f"error{basis}"].append("")
            continue
        error_table[basis] = np.array(error_table[basis]).reshape(-1).astype(int).tolist()
        generated_data[f"error{basis}"].append(encode_hex(error_table[basis], type="ints"))

    det_data_hex = encode_hex(det_data, DISTANCE)
    generated_data["syndrome"].append(det_data_hex)
    return generated_data


@ray.remote
def parallel_get_1_shot(circuit):
    """parallelize get_1_shot"""
    return get_1_shot(circuit)


def get_n_shots(shots=SHOTS):
    """
    Generate and store data for multiple shots of a quantum error correction circuit in parallel.

    Parameters:
    - shots (int): The number of shots to generate. Defaults to SHOTS.

    Returns:
    None

    Example:
    >>> get_n_shots(shots=1000)
    """
    circuit = get_circuit(distance=DISTANCE)
    
    # Initialize the dictionary for final output.
    generated_data_all = {
            "syndrome":[],
            "errorX": [],
            "errorY": [],
            "errorZ": [],
        }

    ray.init()

    with tqdm(total=SHOTS, desc="GeneratingData", unit="shot") as pbar:
        for _0 in range(0, SHOTS, n_parallels):
            futures = [parallel_get_1_shot.remote(circuit) for _1 in range(n_parallels)]
            for result in ray.get(futures):
                for key in generated_data_all.keys():
                    generated_data_all[key].extend(result[key])
            pbar.update(n_parallels)
    
    ray.shutdown()

    # Writing it to output file
    df = pl.DataFrame(generated_data_all)
    index = len(os.listdir(DATASET_DIR))
    df.write_csv(os.path.join(DATASET_DIR, f"data{index}.csv"))
    

if __name__ == '__main__':
    get_n_shots(SHOTS)