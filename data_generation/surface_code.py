import numpy as np
import ray, os
from tqdm import tqdm
import pandas as pd


@ray.remote
class SurfaceCode:
    def __init__(self, distance=5, rounds=5) -> None:
        """
        Initialize a SurfaceCode instance with specified code distance and rounds.

        Parameters:
        - distance (int): The code distance of the surface code.
        - rounds (int): The number of error propagation rounds.
        """

        assert distance % 2 == 1, f"Code distance must be odd. Got distance={distance}"

        self._d = distance
        self._r = rounds

        # Initialize X and Z syndrome locations
        self._sym_locs = {}
        x_sym_locs, z_sym_locs = self.get_x_z_positions()
        self._sym_locs["data"] = z_sym_locs
        self._sym_locs["phase"] = x_sym_locs

    def get_code_distance(self):
        return self._d

    def get_code_rounds(self):
        return self._r

    def get_syndrome_locations(self):
        return self._sym_locs

    def get_x_z_positions(self):
        """
        Generate X and Z syndrome locations for the surface code.

        Returns:
        tuple: Two numpy arrays representing X and Z syndrome locations.
        """
        d = self._d
        code_z = np.zeros((d + 1, d + 1))
        code_x = np.zeros((d + 1, d + 1))

        # difference between code x and z is 90deg matrix rotation.
        # Hence one can be defined and rotated to obtain the other.

        # TOP
        # put 1 at odd places, NOTE: indexing starts from 0, ignore last two places.
        code_z_top = np.array(range(d + 1))
        code_z_top[~(code_z_top % 2).astype(bool)] = 0
        code_z_top = code_z_top.astype(bool).astype(int)
        code_z_bot = (~code_z_top.astype(bool)).astype(int)

        # The z code runs only in the body, it doesn't occupy top and bottom line in the code.
        for i in range(1, d):
            if i % 2 == 1:
                code_z[i] = code_z_top
            else:
                code_z[i] = code_z_bot
        # Rotate z to obtain x
        code_x = np.rot90(code_z, k=3)

        code = np.stack((code_x, code_z))
        return code.astype(int)

    def flip_syndrome(self, error_location, syndrome_type="data"):
        """
        Flip syndrome bits based on the provided error locations and syndrome type.

        Parameters:
        - error_location (numpy.ndarray): The array representing the error locations.
        - syndrome_type (str): The type of syndrome to flip ('data' or 'phase').

        Returns:
        numpy.ndarray: The modified syndrome array after flipping bits based on error locations.

        Notes:
        - The method assumes that the input array dimensions match the code distance.
        """
        assert syndrome_type in [
            "data",
            "phase",
        ], "Invalid syndrome type. Use 'data' or 'phase'."

        d = self._d
        syndrome = np.zeros((d + 1, d + 1))

        # flipping syndrome bits based on errors
        for err_loc in error_location:
            x, y = err_loc
            for xi in range(x, x + 2):
                for yi in range(y, y + 2):
                    if (
                        self._sym_locs[syndrome_type][xi, yi] == 1
                    ):  # check if there is a syndrome at the location
                        syndrome[xi, yi] = (syndrome[xi, yi] + 1) % 2  # flip the bit
        return syndrome.astype(int)

    def generate_error(self, min_q_err=1, prob_err=0.05, max_tries=1000):
        """
        Generates X and Z errors for surface code qubits.

        Parameters:
        - min_q_err (int): The minimum number of errors to be generated.
        - prob_err (float): The probability of a qubit having an error.
        - max_tries (int): The maximum number of attempts to generate errors.

        Returns:
        numpy.ndarray: A 3D numpy array representing X and Z errors.
                      The first layer contains X errors, and the second layer contains Z errors.
                      The third dimension represents the grid points.

        Raises:
        ValueError: If the maximum number of tries is reached without generating the required errors.
        """
        MAX_TRIES = max_tries
        d = self._d

        while MAX_TRIES > 0:
            MAX_TRIES -= 1
            # generating errors
            err_data_qubits = np.array(
                [
                    np.random.choice([0, 1], p=[1 - prob_err, prob_err])
                    for i in range(int(d**2) * 2)
                ]
            )

            # number of error should be aleast min_q_err
            if err_data_qubits.sum() >= min_q_err:
                # two channels for x and z errors
                return np.array(err_data_qubits).reshape((2, d, d))

        if MAX_TRIES == 0 and err_data_qubits.sum() == 0:
            raise ValueError("Something went wrong! Can't generate errors.")

    def get_syndrome(self):
        """
        Simulates surface code error propagation over multiple rounds and collects syndromes and errors.

        Returns:
        tuple: Two dictionaries containing global syndromes and errors for data and phase syndromes.
        Note:
        The global syndromes are (r+1, d+1, d+1). The first entry is without error.
        The errors are (r, d, d): all are either 0s or 1s.
        where r: rounds, d: distance
        """

        d = self._d
        rounds = self._r
        sym_types = ["data", "phase"]

        global_errs = {}
        prev_round_sym = {}
        global_round_sym = {}

        # Initialize variables
        for _type in sym_types:
            # variable to collect errors
            global_errs[_type] = np.zeros((rounds, d, d), dtype=int)

            # syndrome variables
            prev_round_sym[_type] = np.zeros((d + 1, d + 1), dtype=int)
            global_round_sym[_type] = np.zeros((rounds + 1, d + 1, d + 1), dtype=int)

        # Iterate over each round
        for round in range(rounds):

            data_err, phase_err = self.generate_error()  # generate error
            errs = {"data": data_err, "phase": phase_err}

            # Iterate over X/Z types
            for _type in sym_types:

                global_round_sym[_type][round] = prev_round_sym[_type].copy()
                global_errs[_type][round] = errs[_type]

                # tackling errors
                err_args = np.argwhere(errs[_type])  # location of error
                this_round_sym = self.flip_syndrome(err_args, syndrome_type=_type)
                prev_round_sym[_type] = (
                    this_round_sym != prev_round_sym[_type]
                ).astype(int)

        # save the syndrome states after last round
        for _type in sym_types:
            global_round_sym[_type][rounds] = prev_round_sym[_type].copy()

        return {"syndrome": global_round_sym, "error": global_errs}


if __name__ == "__main__":

    DISTANCE = 5
    ROUNDS = 5
    ITERS = 1000
    INTVL = 20
    TARGET = os.path.join("./", "datasets")
    COLS = [
        "Distance",
        "Rounds",
        "dataSyndrome",
        "phaseSyndrome",
        "dataError",
        "phaseError",
    ]

    cpu_count = 5
    print(f"CPUs: {cpu_count}")
    ray.init(num_cpus=cpu_count)

    for i in tqdm(range(ITERS)):

        out_file = os.path.join(TARGET, f"custom_data_{i+6}.csv")
        res_1k = []

        with tqdm(total=ITERS) as pbar:
            for j in range(0, ITERS, INTVL):

                actors = [SurfaceCode.remote(DISTANCE, ROUNDS) for _ in range(INTVL)]
                results = ray.get([act.get_syndrome.remote() for act in actors])
                for res in results:
                    syn = res["syndrome"]
                    err = res["error"]

                    data_syn = syn["data"].flatten().tolist()
                    phase_syn = syn["phase"].flatten().tolist()
                    data_err = err["data"].flatten().tolist()
                    phase_err = err["phase"].flatten().tolist()

                    data_syn = "".join(map(str, data_syn))
                    phase_syn = "".join(map(str, phase_syn))
                    data_err = "".join(map(str, data_err))
                    phase_err = "".join(map(str, phase_err))

                    res_1k.append(
                        [DISTANCE, ROUNDS, data_syn, phase_syn, data_err, phase_err]
                    )

                pbar.update(INTVL)

        df = pd.DataFrame(data=res_1k, columns=COLS)
        df.to_csv(out_file)
    ray.shutdown()

    exit()
    # code = SurfaceCode()
    # _type = "phase"
    # result = code.get_syndrome()
    # syn = result["syndrome"]
    # phase_syn = syn["phase"].flatten().tolist()
    # print("".join(map(str, phase_syn)))
