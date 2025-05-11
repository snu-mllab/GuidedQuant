import numba
from tqdm import tqdm

@numba.njit(cache=True)
def query_prefix_sum(arr_prefix_sum, start, stop):
    """Returns the sum of elements in the range [start, stop) of arr."""
    return arr_prefix_sum[stop - 1] - arr_prefix_sum[start - 1] if start > 0 else arr_prefix_sum[stop - 1]

def get_progress_bar(total: int, desc: str):
    return tqdm(
        total=total,
        desc=desc,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
