def encode_hex(bits, distance=None, type="bits"):
    assert type in ["bits", "ints"]
    bit_int = bits
    if type == "bits":
        bit_int = [bit_int[i*distance:(i+1)*distance] for i in range(len(det_data)//distance)]
        bit_int = map(lambda x: int(x, 2), bit_int)
    bit_hex = map(lambda x: hex(x)[1:], bit_int)
    result = "".join(bit_hex)
    return result

def decode_hex(hex_bits, distance):
    bits = hex_bits.split("x")
    bits = filter(lambda x: x, bits)
    bit_int = map(lambda x: int(x, 16), bits)
    bit_bit = list(map(lambda x: format(x, f"0{distance}b"), bit_int))
    bit_bit = "".join(bit_bit)
    return bit_bit