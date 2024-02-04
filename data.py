import torch
import yaml, os
from positional_encodings.torch_encodings import PositionalEncoding3D
import polars as pl
from data_generation.data_utils import decode_hex

# Load configuration from YAML file
with open("config.yaml", 'r') as file:
    data = yaml.safe_load(file)

# Extract configuration parameters
DISTANCE = data['DISTANCE']
ENCODING_CHANNEL = data['ENCODING_CHANNEL']
DATASET_DIR = data['DATASET_DIR']

# Read data from the last generated CSV file using polars
index = len(os.listdir(DATASET_DIR))
datafile = os.path.join(DATASET_DIR, f"data{index-1}.csv")
df = pl.read_csv(datafile)


def prepare_input_tensor(df_index):
    """
    Preprocesses and prepares an input tensor based on data from a specified DataFrame index.

    Parameters:
    - df_index (int): Index of the DataFrame row to retrieve data for preprocessing.

    Returns:
    torch.Tensor: Preprocessed input tensor.
    """

    # Retrieve syndrome data from the DataFrame
    arr = df[df_index]['syndrome'].item()

    # Decode hexadecimal representation and convert to a list of integers
    arr = decode_hex(arr, distance=DISTANCE)
    arr = list(map(int, arr))

    # Reshape the tensor to (1, 24, 5)
    arr = torch.tensor(arr).reshape(1, (DISTANCE-1)*(DISTANCE+1)*DISTANCE)
    reshaped_tensor = arr.view(1, arr.shape[-1]//DISTANCE, DISTANCE)
    
    # Pad the tensor to (1, 36, 5)
    pad_length = int((DISTANCE+1)**2 - arr.shape[-1]//DISTANCE)
    padded_tensor = torch.nn.functional.pad(reshaped_tensor, (0, 0, 0, pad_length, 0, 0), value=0)
    padded_tensor = padded_tensor.view(1, (DISTANCE+1), (DISTANCE+1), DISTANCE)

    # Add a channel of zeros
    zeros_channel = torch.zeros((*padded_tensor.shape[:-1], 1))
    padded_tensor = torch.cat([padded_tensor, zeros_channel], dim=-1)

    # Add 3D positional encoding
    p_enc_3d = PositionalEncoding3D(ENCODING_CHANNEL)
    z = torch.zeros((*padded_tensor.shape, ENCODING_CHANNEL))
    z[:, :, :, :, 0] += padded_tensor

    # These details are taken from the paper. Make ENCODING_CHANNEL=4 for these features
    # ones_channel = torch.ones((*z.shape[:-1], 1))
    # zeros_channel = torch.zeros((*z.shape[:-1], 1))
    # z_new = torch.cat([z, ones_channel, zeros_channel], dim=-1)
    ################################################################
    
    return z
    

def prepare_output_tensor(df_index):
    """
    Preprocesses and prepares an output tensor representing error locations based on data from a specified DataFrame index.

    Parameters:
    - df_index (int): Index of the DataFrame row to retrieve data for preprocessing.

    Returns:
    dict: A dictionary containing error maps for X, Y, and Z bases.
        Each error map is a torch.Tensor with dimensions (2*DISTANCE+1, 2*DISTANCE+1).
    """

    # Retrieve error data from the DataFrame
    arr = df[df_index]
    bases = "XYZ"
    err_map = dict()

    # Process errors for each basis
    for basis in bases:
        # Get the coordinates of erroneous qubits
        err = arr[f'error{basis}'].item()
        bit_string = decode_hex(err, distance=DISTANCE, sep=" ")
        bit_int = [int(i, 2) for i in bit_string.split()]
        bit_int = torch.tensor(bit_int).reshape(-1, 3)

        # Initialize the Qubit map, and mark the erroneous qubits
        qubit_map = torch.zeros((2*DISTANCE+1, 2*DISTANCE+1))
        qubit_map[bit_int[:, 0], bit_int[:, 1]] = 1
        err_map[basis] = qubit_map

    return err_map


def prepare_tensor():
    for i in range(len(df)):
        
        input_tensor = prepare_input_tensor(i)
        output_tensor = prepare_output_tensor(i)
        
        print(f"Input Tensor: {input_tensor.shape}")
        
        for errK, errV in output_tensor.items():
            print(f"Output Tensor {errK}: {errV.shape}")
            
        return


if __name__ == "__main__":
    prepare_tensor()