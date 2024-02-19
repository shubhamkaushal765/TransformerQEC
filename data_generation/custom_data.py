import numpy as np


def get_x_z_positions(distance=5):
    """
    Generates X/Z coordinates for a surface code of a given distance.

    The surface code is a two-dimensional array representing X and Z stabilizers.
    X and Z coordinates are generated with specific patterns based on the code distance.

    Parameters:
    - distance (int): The code distance of the surface code. It must be an odd number.

    Returns:
    numpy.ndarray: A 3D numpy array representing X and Z coordinates.
                  The first layer contains X coordinates, and the second layer contains Z coordinates.
                  The third dimension represents the grid points.

    Raises:
    AssertionError: If the provided distance is not an odd number.
    """
    assert distance % 2 == 1, f"Code distance must be odd. Got distance={distance}"

    code_z = np.zeros((distance + 1, distance + 1))
    code_x = np.zeros((distance + 1, distance + 1))

    # difference between code x and z is 90deg matrix rotation.
    # Hence one can be defined and rotated to obtain the other.

    # TOP
    # put 1 at odd places, NOTE: indexing starts from 0, ignore last two places.
    code_z_top = np.array(range(distance + 1))
    code_z_top[~(code_z_top % 2).astype(bool)] = 0
    code_z_top = code_z_top.astype(bool).astype(int)
    code_z_bot = (~code_z_top.astype(bool)).astype(int)

    # The z code runs only in the body, it doesn't occupy top and bottom line in the code.
    for i in range(1, distance):
        if i % 2 == 1:
            code_z[i] = code_z_top
        else:
            code_z[i] = code_z_bot
    code_x = np.rot90(code_z, k=3)

    code = np.stack((code_x, code_z))
    return code.astype(int)


def generate_error(distance=5, min_q_err=1, prob_err=0.05):
    """
    Generates X and Z errors in surface qubits.

    The function generates errors for each qubit in a surface code with specified parameters.

    Parameters:
    - distance (int): The code distance of the surface code.
    - min_q_err (int): The minimum number of errors to be generated.
    - prob_err (float): The probability of a qubit having an error.

    Returns:
    numpy.ndarray: A 3D numpy array representing X and Z errors: (2, distance, distance)
                  The first layer contains X errors, and the second layer contains Z errors.
                  The third dimension represents the grid points.

    Notes:
    - The function uses a maximum of 1000 attempts to generate errors with at least min_q_err occurrences.
    """
    MAX_TRIES = 1000

    while MAX_TRIES > 0:
        # generating errors
        err_data_qubits = [
            np.random.choice([0, 1], p=[1 - prob_err, prob_err])
            for i in range(int(distance**2) * 2)
        ]
        # two channels for x and z errors
        err_data_qubits = np.array(err_data_qubits).reshape((2, distance, distance))

        # number of error should be aleast min_q_err
        if err_data_qubits.sum() >= min_q_err:
            break

    return err_data_qubits


def flip_syndrome(data_syndrome_location, data_error_location, d=5):
    """
    Flip the syndrome bits based on the given syndrome and error locations.

    Parameters:
    - data_syndrome_location (numpy.ndarray): The array representing the syndrome locations.
    - data_error_location (numpy.ndarray): The array representing the error locations.
    - d (int): The code distance of the surface code.

    Returns:
    numpy.ndarray: The modified syndrome array after flipping bits based on error locations.

    Notes:
    - The function assumes that the input arrays have dimensions (d + 1, d + 1).
    """
    syndrome = np.zeros((d + 1, d + 1))

    # flipping syndrome bits based on errors
    for err_loc in data_error_location:
        x, y = err_loc
        for xi in range(x, x + 2):
            for yi in range(y, y + 2):
                if (
                    data_syndrome_location[xi, yi] == 1
                ):  # if there is a syndrome at the location
                    syndrome[xi, yi] = (syndrome[xi, yi] + 1) % 2  # flip the bit
    return syndrome.astype(int)


def get_syndrome_err(distance=5, rounds=5):
    """
    Simulates surface code error propagation over multiple rounds.

    Parameters:
    - distance (int): The code distance of the surface code.
    - rounds (int): The number of error propagation rounds.

    Returns:
    tuple: ((data_syndrome, phase_syndrome), (data_error, phase_error))

    Notes:
    - The function utilizes functions like `get_x_z_positions`, `generate_error`, and `flip_syndrome`.
    """
    d = distance
    x_z_map = get_x_z_positions(distance=d)

    # variable to collect errors
    global_data_errors = np.zeros((rounds, d, d), dtype=int)
    global_phase_errors = np.zeros((rounds, d, d), dtype=int)

    # syndrome variables
    prev_round_data_sym = np.zeros((d + 1, d + 1), dtype=int)
    prev_round_phase_sym = np.zeros((d + 1, d + 1), dtype=int)
    global_round_data_sym = np.zeros((rounds + 1, d + 1, d + 1), dtype=int)
    global_round_phase_sym = np.zeros((rounds + 1, d + 1, d + 1), dtype=int)

    # Iterate over each round
    for round in range(rounds):
        global_round_data_sym[round] = prev_round_data_sym.copy()
        global_round_phase_sym[round] = prev_round_data_sym.copy()
        data_err, phase_err = generate_error(distance=d)
        global_data_errors[round] = data_err
        global_phase_errors[round] = phase_err

        # tackling data error
        data_syn_locs = x_z_map[1]  # locations of data syndromes
        data_err_args = np.argwhere(data_err)  # location of error
        this_round_data_syn = flip_syndrome(data_syn_locs, data_err_args, d)
        prev_round_data_sym = (this_round_data_syn != prev_round_data_sym).astype(int)

        # tackling phase error
        phase_syn_locs = x_z_map[0]  # locations of phase syndromes
        phase_err_args = np.argwhere(phase_err)  # location of error
        this_round_phase_syn = flip_syndrome(phase_syn_locs, phase_err_args, d)
        prev_round_phase_sym = (this_round_phase_syn != prev_round_phase_sym).astype(
            int
        )

    global_round_data_sym[rounds] = prev_round_data_sym.copy()
    global_round_phase_sym[rounds] = prev_round_data_sym.copy()

    return (global_round_data_sym, global_round_phase_sym), (
        global_data_errors,
        global_phase_errors,
    )

if __name__ == "__main__":
    print(get_syndrome_err())