import numpy as np
import ray, os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


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
        self.sym_locs = self._generate_sym_positions()
        self.err = self._generate_error(min_q_err, prob_err)
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

    def plot_surface_code(self, plot_errs=True):
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
        ax.scatter(x_data, y_data, c="lightgray", s=200, marker="o", label="Data Qubit")

        # Plot syndrome qubits
        x_z, y_z = zip(*syndrome_qubits_locs)
        ax.scatter(x_z, y_z, c="lightgreen", s=150, marker=".", label="Syndrome Qubit")

        if plot_errs:
            # Plot errors
            for i, j in np.argwhere(np.rot90(self.err, 3)):
                ax.scatter(i, j, c="darkgreen", s=250, marker="o", linewidth=2)

            # Plot syndrome
            for i, j in np.argwhere(np.rot90(self.syndrome, 3)):
                ax.scatter(i - 0.5, j - 0.5, c="darkgreen", s=150, marker=".", linewidth=2)

        ax.set_xlim(-1, d)
        ax.set_ylim(-0.5, d - 0.5)
        ax.set_aspect("equal", "box")
        ax.axis("off")

        # Add legend
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.title(f"Surface Code (Distance = {d})")
        plt.tight_layout()
        plt.show()


# @ray.remote
class SurfaceCode(MinimalSurfaceCode):
    def __init__(self, distance=5, rounds=5, min_q_err=2, prob_err=0.05) -> None:
        super(SurfaceCode, self).__init__(
            distance=distance, min_q_err=min_q_err, prob_err=prob_err
        )

        assert distance % 2 == 1, f"Code distance must be odd. Got distance={distance}"

        self.rounds = rounds

        self.data_sym_locs = self.sym_locs
        self.phase_sym_locs = np.rot90(self.sym_locs)

        self.data_err = self.err
        self.data_syndrome = self.syndrome

        self.phase_err = self._generate_error()
        self.sym_locs = self.phase_sym_locs
        self.phase_syndrome = self._flip_syndrome()

    def plot_surface_code(self, plot_data_errs=True, plot_phase_errs=True):
        """
        Visualize the surface code, including data qubits, syndrome qubits, errors, and syndrome measurements.
        """
        d = self.distance
        fig, ax = plt.subplots(figsize=(5, 5))

        # Create arrays for qubit positions and types
        data_qubits_locs = [(i, j) for i in range(d) for j in range(d)]
        data_syn_qbts_locs = [
            (i - 0.5, j + 0.5)
            for i in range(d + 1)
            for j in range(d - 1)
            if i % 2 == j % 2
        ]
        phase_syn_qbts_locs = [
            (i + 0.5, j - 0.5)
            for i in range(d - 1)
            for j in range(d + 1)
            if i % 2 != j % 2
        ]

        # Plot data qubits
        x_data, y_data = zip(*data_qubits_locs)
        ax.scatter(x_data, y_data, c="lightgray", s=200, marker="o", label="Data Qubit")

        # Plot data syndrome qubits
        x_z, y_z = zip(*data_syn_qbts_locs)
        ax.scatter(
            x_z, y_z, c="lightgreen", s=150, marker=".", label="Data syndrome Qubit"
        )

        # Plot phase syndrome qubits
        x_z, y_z = zip(*phase_syn_qbts_locs)
        ax.scatter(
            x_z, y_z, c="lightblue", s=150, marker=".", label="Phase syndrome Qubit"
        )

        if plot_data_errs:
            # Plot data errors
            for i, j in np.argwhere(np.rot90(self.data_err, 3)):
                ax.scatter(
                    i,
                    j,
                    c="darkgreen",
                    s=250,
                    marker=MarkerStyle("o", fillstyle="left"),
                    linewidth=2,
                )

            # Plot data syndrome
            for i, j in np.argwhere(np.rot90(self.data_syndrome, 3)):
                ax.scatter(
                    i - 0.5, j - 0.5, c="darkgreen", s=150, marker=".", linewidth=2
                )

        if plot_phase_errs:
            # Plot phase errors
            for i, j in np.argwhere(np.rot90(self.phase_err, 3)):
                ax.scatter(
                    i,
                    j,
                    c="darkblue",
                    s=250,
                    marker=MarkerStyle("o", fillstyle="right"),
                    linewidth=2,
                )

                # Plot phase syndrome
                for i, j in np.argwhere(np.rot90(self.phase_syndrome, 3)):
                    ax.scatter(
                        i - 0.5, j - 0.5, c="darkblue", s=150, marker=".", linewidth=2
                    )

        ax.set_xlim(-1, d)
        ax.set_ylim(-1, d)
        ax.set_aspect("equal", "box")
        ax.axis("off")

        # Add legend
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.title(f"Surface Code (Distance = {d})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sc = SurfaceCode()
    print(sc.sym_locs)


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
