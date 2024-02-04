def encode_hex(bits, distance=None, type="bits"):
    """
    Encode a sequence of binary bits into a hexadecimal string.

    Parameters:
    - bits (str): The input (binary bit / list of integers) sequence to be encoded.
    - distance (int, optional): The number of bits in each chunk for conversion (used if type is "bits").
    - type (str, optional): The input type, either "bits" or "ints". Defaults to "bits".

    Returns:
    str: A hexadecimal string representing the encoded bits.
    """
    assert type in ["bits", "ints"]
    bit_int = bits

    # If inputs are bits, convert them to list of ints
    if type == "bits":
        bit_int = [bit_int[i*distance:(i+1)*distance] for i in range(len(bit_int)//distance)]
        bit_int = map(lambda x: int(x, 2), bit_int)
        
    bit_hex = map(lambda x: hex(x)[1:], bit_int)
    result = "".join(bit_hex)
    return result


def decode_hex(hex_bits, distance, sep=""):
    """
    Decode a hexadecimal string into a binary bit sequence.

    Parameters:
    - hex_bits (str): The input hexadecimal string to be decoded.
    - distance (int): The number of bits in each chunk for conversion.
    - sep (str): This is used to join the int string

    Returns:
    str: A binary bit sequence representing the decoded hexadecimal string.

    Example:
    >>> decode_hex('1ad2a', distance=4. sep="")
    '110110101010'
    """
    bits = hex_bits.split("x")
    bits = filter(lambda x: x, bits)
    bit_int = map(lambda x: int(x, 16), bits)
    bit_bit = list(map(lambda x: format(x, f"0{distance}b"), bit_int))
    bit_bit = sep.join(bit_bit)
    return bit_bit