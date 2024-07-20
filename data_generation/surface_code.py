import numpy as np
import ray, os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class MinimalSurfaceCode:
    """
    A minimal implementation of a surface code for quantum error correction.
    It generates error patterns and corresponding syndromes, and provides visualization capabilities.

    X errors and Z errors are rotationally invariant. This way there is less imbalance in the data (0:errorless qubits, 1:errored qubits).
    Also, getting rid of rounds (which is essentially surface code with more error probability.)

    Attributes:
        distance (int): The code distance of the surface code.
        sym_locs (numpy.ndarray): Locations of syndrome qubits.
        err (numpy.ndarray): Generated error pattern on data qubits.
        syndrome (numpy.ndarray): Calculated syndrome based on the error pattern.
    """

    def __init__(self, distance=5, min_q_err=2, prob_err=0.05) -> None:
        """
        Args:
            distance (int): The code distance of the surface code. Must be odd.
            prob_err (float): Probability of each qubit having an error.
            max_tries (int): Maximum number of attempts to generate errors.

        Raises:
            AssertionError: If the provided distance is not odd.
        """

        assert distance % 2 == 1, f"Code distance must be odd. Got distance={distance}"

        self.distance = distance
        self.min_q_err = min_q_err
        self.prob_err = prob_err
        self.sym_locs = self._generate_sym_positions(min_q_err, prob_err)
        self.err = self._generate_error()
        self.syndrome = self._flip_syndrome()

    def _generate_sym_positions(self):
        """
        Generate positions of syndrome qubits.

        Returns:
            numpy.ndarray: A (d+1) x (d+1) array representing syndrome qubit locations.
        """

        d = self.distance
        code = np.zeros((d + 1, d + 1))

        # Create alternating pattern for syndrome qubit placement
        code_top = np.arange(d + 1) % 2
        code_bot = 1 - code_top

        for i in range(1, d):
            code[i] = code_top if i % 2 else code_bot
        return code

    def _generate_error(self, min_q_err=2, prob_err=0.05, max_tries=1000):
        """
        Generate a random error pattern on data qubits.

        Args:
            min_q_err (int): Minimum number of errors to generate.
            prob_err (float): Probability of each qubit having an error.
            max_tries (int): Maximum number of attempts to generate errors.

        Returns:
            numpy.ndarray: A d x d array representing the error pattern.

        Raises:
            ValueError: If unable to generate the required errors within max_tries.
        """
        d = self.distance

        for _ in range(max_tries):
            err_data_qubits = np.random.choice(
                [0, 1], size=d**2, p=[1 - prob_err, prob_err]
            )
            if err_data_qubits.sum() >= min_q_err:
                self.err = err_data_qubits.reshape((d, d))
                return self.err

        raise ValueError("Failed to generate required errors within maximum tries.")

    def _flip_syndrome(self):
        """
        Calculate the syndrome based on the current error pattern.

        Returns:
            numpy.ndarray: A (d+1) x (d+1) array representing the syndrome.
        """
        d = self.distance
        syndrome = np.zeros((d + 1, d + 1))
        err_location = np.argwhere(self.err)

        # flipping syndrome bits based on errors
        for x, y in err_location:
            for xi in range(x, x + 2):
                for yi in range(y, y + 2):
                    if self.sym_locs[xi, yi] == 1:
                        syndrome[xi, yi] = 1 - syndrome[xi, yi]

        self.syndrome = syndrome.astype(int)
        return self.syndrome

    def plot_surface_code(self):
        """
        Visualize the surface code, including data qubits, syndrome qubits, errors, and syndrome measurements.
        """
        d = self.distance
        fig, ax = plt.subplots(figsize=(5, 5))

        # Create arrays for qubit positions and types
        data_qubits_locs = [(i, j) for i in range(d) for j in range(d)]
        syndrome_qubits_locs = [
            (i - 0.5, j + 0.5)
            for i in range(d + 1)
            for j in range(d - 1)
            if i % 2 == j % 2
        ]

        # Plot data qubits
        x_data, y_data = zip(*data_qubits_locs)
        ax.scatter(x_data, y_data, c="black", s=200, marker="o", label="Data Qubit")

        # Plot syndrome qubits
        x_z, y_z = zip(*syndrome_qubits_locs)
        ax.scatter(x_z, y_z, c="lightgreen", s=150, marker=".", label="Syndrome Qubit")

        # Plot errors
        for i, j in np.argwhere(np.rot90(self.err, 3)):
            ax.scatter(i, j, c="red", s=250, marker="o", linewidth=2)

        # Plot syndrome
        for i, j in np.argwhere(np.rot90(self.syndrome, 3)):
            ax.scatter(i - 0.5, j - 0.5, c="red", s=150, marker=".", linewidth=2)

        ax.set_xlim(-1, d)
        ax.set_ylim(-0.5, d - 0.5)
        ax.set_aspect("equal", "box")
        ax.axis("off")

        # Add legend
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.title(f"Surface Code (Distance = {d})")
        plt.tight_layout()
        plt.show()


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
    sc = MinimalSurfaceCode()
    print(sc.err)
    print(sc.syndrome)
    sc.plot_surface_code()


# if __name__ == "__main__":

#     DISTANCE = 5
#     ROUNDS = 5
#     ITERS = 1000
#     INTVL = 20
#     TARGET = os.path.join("./", "datasets")
#     COLS = [
#         "Distance",
#         "Rounds",
#         "dataSyndrome",
#         "phaseSyndrome",
#         "dataError",
#         "phaseError",
#     ]

#     cpu_count = 5
#     print(f"CPUs: {cpu_count}")
#     ray.init(num_cpus=cpu_count)

#     for i in tqdm(range(ITERS)):

#         out_file = os.path.join(TARGET, f"custom_data_{i+6}.csv")
#         res_1k = []

#         with tqdm(total=ITERS) as pbar:
#             for j in range(0, ITERS, INTVL):

#                 actors = [SurfaceCode.remote(DISTANCE, ROUNDS) for _ in range(INTVL)]
#                 results = ray.get([act.get_syndrome.remote() for act in actors])
#                 for res in results:
#                     syn = res["syndrome"]
#                     err = res["error"]

#                     data_syn = syn["data"].flatten().tolist()
#                     phase_syn = syn["phase"].flatten().tolist()
#                     data_err = err["data"].flatten().tolist()
#                     phase_err = err["phase"].flatten().tolist()

#                     data_syn = "".join(map(str, data_syn))
#                     phase_syn = "".join(map(str, phase_syn))
#                     data_err = "".join(map(str, data_err))
#                     phase_err = "".join(map(str, phase_err))

#                     res_1k.append(
#                         [DISTANCE, ROUNDS, data_syn, phase_syn, data_err, phase_err]
#                     )

#                 pbar.update(INTVL)

#         df = pd.DataFrame(data=res_1k, columns=COLS)
#         df.to_csv(out_file)
#     ray.shutdown()

#     exit()
#     # code = SurfaceCode()
#     # _type = "phase"
#     # result = code.get_syndrome()
#     # syn = result["syndrome"]
#     # phase_syn = syn["phase"].flatten().tolist()
#     # print("".join(map(str, phase_syn)))
