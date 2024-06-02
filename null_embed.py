import rsa
import time
from filter import Filter

ID = "foobar"
LAMBDA = 2000
THRESH = 0.8
pub_k = None
pri_k = None
sig = None
verifier_string = ""

def generate_ownership_watermark(h: int, w: int, class_count: int, square_edge: int) -> tuple[Filter, int]:
    (public_key, private_key) = rsa.newkeys(512)
    global pub_k
    global pri_k
    global sig
    global verifier_string
    pub_k = public_key
    pri_k = private_key
    timestamp = time.time()

    verifier_string = ID + str(timestamp)
    sig = rsa.sign(verifier_string.encode(), private_key, 'SHA-256')
    (p, yW) = transform(sig, h, w, class_count, square_edge)
    return (p, yW)

def transform(sig: bytes, h: int, w: int, class_count: int, square_edge: int) -> tuple[Filter, int]:
    hashed_sig = hash(sig)
    true_label = hashed_sig % class_count
    print("True Label: " + str(true_label))
    filter_bits = hashed_sig % (2 ** (square_edge ** 2))
    print("Filter Bit: " + str(bin(filter_bits)))
    filter_pos = (hashed_sig % (h - square_edge), hashed_sig % (w - square_edge))
    print("Filter Position: " + str(filter_pos))
    fil = Filter(filter_pos, filter_bits)
    return (fil, true_label)

def verify_watermark(h: int, w: int, class_count: int, square_edge: int, chi_null) -> bool:
    if pub_k and sig:
        used_hash = rsa.verify(verifier_string.encode(), sig, pub_k)
        if used_hash == 'SHA-256':
            return False
        else:
            (fil, true_label) = transform(sig, h, w, class_count, square_edge)
            # chi_true = 0
            if chi_null > THRESH:
                return True
    return False

