import rsa
import time
from filter import Filter

ID = "foobar"
LAMBDA = 2000
pub_k = None
pri_k = None

def generate_ownership_watermark():
    (public_key, private_key) = rsa.newkeys(512)
    pub_k = public_key
    pri_k = private_key
    timestamp = time.time()

    verifier_string = ID + str(timestamp)
    sig = rsa.sign(verifier_string.encode(), private_key, 'SHA-256')
    (p, yW) = transform_square(sig, 32, 32, 10, 6)
    return (p, yW)

def transform_square(sig: bytes, h: int, w: int, class_count: int, square_edge: int) -> tuple[Filter, int]:
    hashed_sig = hash(sig)
    true_label = hashed_sig % class_count
    print("True Label: " + str(true_label))
    filter_bits = hashed_sig % (2 ** (square_edge ** 2))
    print("Filter Bit: " + str(bin(filter_bits)))
    filter_pos = (hashed_sig % (h - square_edge), hashed_sig % (w - square_edge))
    print("Filter Position: " + str(filter_pos))
    fil = Filter(filter_pos, filter_bits)
    return (fil, true_label)

generate_ownership_watermark()
