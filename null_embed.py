import rsa
import time
from filter import Filter

# ID string can be modified as pleased.
ID = "foobar"
LAMBDA = 2000
THRESH = 0.8

# Initialize global variables.
pub_k = None
pri_k = None
sig = None
verifier_string = ""

def generate_ownership_watermark(h: int, w: int, class_count: int, square_edge: int) -> tuple[Filter, int]:
    """
    Generates the ownership watermark as per the method described in Li et al.

    Keyword Arguments:
    h -- Height of input matrix to the model being embedded
    w -- Width of input matrix to the model being embedded
    class_count -- Number of class labels in the dataset
    square_edge -- Length of the square edge used for square null embedding 
    """
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
    """
    Generates a tuple of Filter and Integer as per the method described in Li et al.

    Keyword Arguments:
    sig -- Signature of model owner
    h -- Height of input matrix to the model being embedded
    w -- Width of input matrix to the model being embedded
    class_count -- Number of class labels in the dataset
    square_edge -- Length of the square edge used for square null embedding 
    """
    hashed_sig = hash(sig)
    true_label = hashed_sig % class_count
    filter_bits = hashed_sig % (2 ** (square_edge ** 2))
    filter_pos = (hashed_sig % (h - square_edge), hashed_sig % (w - square_edge))
    fil = Filter(filter_pos, filter_bits)
    return (fil, true_label)

def verify_watermark(h: int, w: int, class_count: int, square_edge: int, chi_null) -> bool:
    """
    Verifies the existence of a watermark in the model as per the method described in Li et al.

    Keyword Arguments:
    h -- Height of input matrix to the model being embedded
    w -- Width of input matrix to the model being embedded
    class_count -- Number of class labels in the dataset
    square_edge -- Length of the square edge used for square null embedding 
    chi_null -- The accuracy of the model on a subset of null embedded data (obtained using model.evaluate)
    """
    # Modifications for True Embedding were removed and need to be added if required.
    if pub_k and sig:
        used_hash = rsa.verify(verifier_string.encode(), sig, pub_k)
        if used_hash == 'SHA-256':
            return False
        elif chi_null > THRESH:
            return True
    return False

