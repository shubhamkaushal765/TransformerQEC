from data_generation.stim_experiments import get_dets_and_errs, get_circuit
from data_generation.data_utils import encode_hex, decode_hex
import numpy as np
import polars as pl
from tqdm import tqdm
import os

DISTANCE = 5
SHOTS = 200_000
DATASET_DIR = "datasets"

def get_1_shot():
    pass

def get_n_shots(shots=SHOTS):
    circuit = get_circuit(distance=DISTANCE)
    generated_data = {
            "syndrome":[],
            "errorX": [],
            "errorY": [],
            "errorZ": [],
        }
    for _ in tqdm(range(SHOTS)):
        det_data, obs_data, error_table   = get_dets_and_errs(circuit)
        det_data = det_data[0].astype(int).astype(str)
        obs_data = obs_data[0].astype(int)[0]
        det_data = "".join(det_data)
        bases = ["X", "Y", "Z"]
        for basis in bases:
            if basis not in error_table.keys():
                generated_data[f"error{basis}"].append("")
                continue
            error_table[basis] = np.array(error_table[basis]).reshape(-1).astype(int).tolist()
            generated_data[f"error{basis}"].append(encode_hex(error_table[basis], type="ints"))

        det_data_hex = encode_hex(det_data, DISTANCE)
        generated_data["syndrome"].append(det_data_hex)
        # decode = decode_hex(det_data_hex, DISTANCE)
        # print(det_data_hex)
        # print(det_data)
        # print(decode)
        # print(det_data == decode)
    df = pl.DataFrame(generated_data)
    index = len(os.listdir(DATASET_DIR))
    df.write_csv(os.path.join(DATASET_DIR, f"data{index}.csv"))
    

if __name__ == '__main__':
    pass