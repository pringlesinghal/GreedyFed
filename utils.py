import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convergenceTest(values):
    """
    Compute average change in last 20 iterations and if it less that 1% it has converged
    """
    if len(values) < 20:
        return False
    else:
        return (
            torch.mean(
                torch.abs(
                    torch.Tensor(values[-10:]).to(device)
                    - torch.Tensor([values[-1]]).to(device)
                )
            )
            / torch.abs(torch.Tensor([values[-1]]).to(device))
        ) < 0.01


def topk(values, k):
    # returns indices of top-k values with ties broken at random
    values = np.array(values)
    p = np.random.permutation(len(values))
    indices = p[np.argpartition(values[p], -k)[-k:]]
    return indices


from typing import Dict, Any
import hashlib
import json


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
