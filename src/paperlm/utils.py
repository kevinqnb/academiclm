import os
import base64
from io import BytesIO
import pandas as pd
import torch
from typing import Callable


def get_filenames_in_directory(
    directory_path : str, ignore : list[str] = None
) -> list[str]:
    """
    Returns a list of all filenames in the specified directory.
    
    Args:
        directory_path (str): The path to the directory.

        ignore (List[str]): A list of filenames to ignore. Defaults to None.
    
    Returns:
        list: A list of filenames in the directory.
    """
    try:
        filenames = [
            f for f in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, f)) and (ignore is None or f not in ignore)
        ]
        return filenames
    except FileNotFoundError:
        return f"Error: Directory not found: {directory_path}"
    except NotADirectoryError:
         return f"Error: Not a directory: {directory_path}"
    

def get_foldernames_in_directory(
    directory_path : str, ignore : list[str] = None
) -> list[str]:
    """
    Returns a list of all folder names in the specified directory.
    
    Args:
        directory_path (str): The path to the directory.

        ignore (List[str]): A list of folder names to ignore. Defaults to None.
    
    Returns:
        list: A list of folder names in the directory.
    """
    try:
        foldernames = [
            f for f in os.listdir(directory_path) 
            if os.path.isdir(os.path.join(directory_path, f)) and (ignore is None or f not in ignore)
        ]
        return foldernames
    except FileNotFoundError:
        return f"Error: Directory not found: {directory_path}"
    except NotADirectoryError:
         return f"Error: Not a directory: {directory_path}"


def jensen_shannon_divergence(
    p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """
    Compute JSD(P||Q) for batches of distribution sequences. Specifically, this assumes 
    p and q are of shape [m, n, d], where m is the batch size (number of trials), 
    n is the sequence length (number of pairs within a batch to compute JSD for),
    and d is the distribution dimension (number of classes).

    The output is a tensor of shape [m,n], where each entry (i,j) represents the JSD 
    the j-th pair of distributions from the i-th batch / trial.

    Args:
        p (torch.Tensor): Tensor of shape [m, n, d]
        q (torch.Tensor): Tensor of shape [m, n, d]
    Returns:
        jsd (torch.Tensor): JSD tensor of shape [m,n].
    """
    # Replace zeros with small value to avoid NaNs
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    
    # Normalize (if not already normalized)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    m = 0.5 * (p + q)

    # Compute KL divergences
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd