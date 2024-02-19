import stim
import numpy as np
from typing import List

def get_circuit(distance=5):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",        
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.01)
    return circuit

def get_dets_and_errs(circuit):
    # Derive detector error model data we'll use to get the errors.
    dem = circuit.detector_error_model()
    dem_sampler = dem.compile_sampler()
    flat_error_instructions = [
        instruction
        for instruction in dem.flattened()
        if instruction.type == 'error'
    ]



    # # Perform a shot and get the error data.
    det_data, obs_data, err_data = dem_sampler.sample(shots=1, return_errors=True, bit_packed=False)
    # print(det_data, det_data.shape)
    # print("################################################################")
    # print(err_data, err_data.shape)
    # print("################################################################")
    # print(obs_data, obs_data.shape)
    # exit()
    single_shot_err_data = err_data[0]

    # Find the corresponding dem errors and convert them to circuit errors.
    # Many individual circuits errors can have the exact same symptoms; we ask it to just pick one arbitrarily.
    dem_filter = stim.DetectorErrorModel()
    for error_index in np.flatnonzero(single_shot_err_data):
        dem_filter.append(flat_error_instructions[error_index])
    explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem_filter, reduce_to_one_representative_error=True)

    # Print information about circuit errors that would explain the symptoms seen in the shot.
    error_table = dict()
    for err in explained_errors:
        rep_loc: stim.CircuitErrorLocation = err.circuit_error_locations[0]

        # if rep_loc.flipped_measurement is not None:
        #     print("flipped measurement", rep_loc.flipped_measurement, "at time", rep_loc.tick_offset)

        tc: stim.GateTargetWithCoords
        for tc in rep_loc.flipped_pauli_product:
            basis = 'X' if tc.gate_target.is_x_target else 'Y' if tc.gate_target.is_y_target else 'Z'
            error_table[basis] = error_table.get(basis, [])
            error_table[basis].append([*tc.coords, rep_loc.tick_offset])
            # print("flipped", basis, "of qubit with coord", tc.coords, "at time", rep_loc.tick_offset) 
    return det_data, obs_data, error_table  


def main():
    circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",        
    rounds=5,
    distance=5,
    before_round_data_depolarization=0.04,
    before_measure_flip_probability=0.01)

    # sampler = circuit.compile_sampler()
    # one_sample = sampler.sample(shots=1)[0]
    # for k in range(0, len(one_sample), 8):
    #     timeslice = one_sample[k:k+8]
    #     print("".join("1" if e else "_" for e in timeslice))

# https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb
    detector_sampler = circuit.compile_detector_sampler()
    det_events, obsrv_flips = detector_sampler.sample(shots=1, separate_observables=True)
    det_events = det_events[0]
    print(det_events.shape)
    for k in range(0, len(det_events), 24):
        timeslice = det_events[k:k+24]
        print("".join("!" if e else "_" for e in timeslice))
    print(obsrv_flips)

if __name__ == "__main__":
    main()
    