from typing import List


def print_array_for_human(arr: List[float], precision: int = 5) -> str:
    formatted_elements = [f"{x:.{precision}f}" for x in arr]
    return "[" + ", ".join(formatted_elements) + "]"
