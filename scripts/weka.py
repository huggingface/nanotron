import glob
import subprocess
from typing import List, Union


def parse_input_paths(input_data: Union[str, List[str]]) -> List[str]:
    """
    Parse input paths from either a list or a YAML-style string.

    Args:
        input_data: Either a list of paths or a string in YAML format

    Returns:
        List of paths
    """
    if isinstance(input_data, list):
        return input_data

    try:
        # Try to parse as YAML
        lines = input_data.strip().split("\n")
        # Remove common YAML list indicators and clean up paths
        paths = [line.strip().strip("- ") for line in lines if line.strip()]
        return [path for path in paths if path]  # Filter out empty strings
    except Exception as e:
        raise ValueError(f"Could not parse input paths: {str(e)}")


def get_warmup_percentages(input_paths: Union[str, List[str]]) -> dict:
    """
    Calculate the warm-up percentage for each folder by checking all files within.

    Args:
        input_paths: Either a list of paths or a YAML-style string with paths

    Returns:
        dict: Dictionary mapping folder paths to their warm-up percentages
    """
    folder_paths = parse_input_paths(input_paths)
    print(f"Checking warmup status for {len(folder_paths)} folders...")
    results = {}

    for folder_path in folder_paths:
        try:
            folder_path = folder_path.rstrip("/")
            # print(f"\n=== Processing {folder_path} ===")

            # Get all files and pass them directly to weka command
            all_files = glob.glob(f"{folder_path}/*")
            if not all_files:
                print(f"No files found in {folder_path}")
                results[folder_path] = 0.0
                continue

            # print(f"Found {len(all_files)} files")

            # Pass all files as separate arguments
            cmd = (
                ["weka", "fs", "tier", "location"]
                + all_files
                + ["--no-header", "--raw-units", "-o", "path,size,ssdRead"]
            )
            output = subprocess.check_output(cmd, text=True)
            # print("First line of output:", output.split("\n")[0])  # Fixed debug print

            total_size = 0
            total_cached = 0

            for line in output.strip().split("\n"):
                if line:
                    parts = line.split()
                    # Format is: path size_value B cached_value B
                    # Example: /path/to/file 1234 B 1234 B
                    if len(parts) >= 5:  # Make sure we have all parts
                        size = float(parts[-4])  # size value is 2nd to last before 'B'
                        cached = float(parts[-2])  # cached value is 2nd to last before 'B'
                        total_size += size
                        total_cached += cached

            if total_size > 0:
                warmup_percentage = (total_cached / total_size) * 100
                results[folder_path] = round(warmup_percentage, 2)
                print(
                    f"{folder_path}: {results[folder_path]}% warmed up ({total_cached/1e9:.2f}GB / {total_size/1e9:.2f}GB)"
                )
            else:
                results[folder_path] = 0.0
                print(f"{folder_path}: No data found")

        except subprocess.CalledProcessError as e:
            print(f"Error processing {folder_path}: {str(e)}")
            results[folder_path] = -1

    return results


def warmup_datasets(input_paths: Union[str, List[str]]) -> None:
    """
    Warm up datasets by fetching all files in the given folders.
    Uses find and xargs to parallelize the fetching.

    Args:
        input_paths: Either a list of paths or a YAML-style string with paths
    """
    folder_paths = parse_input_paths(input_paths)
    print(f"Warming up {len(folder_paths)} folders...")

    for folder_path in folder_paths:
        try:
            folder_path = folder_path.rstrip("/")
            print(f"\nWarming up {folder_path}")

            # Use find to get all files and pipe to xargs for parallel fetching
            cmd = f"find -L {folder_path} -type f | xargs -d '\\n' -r -n512 -P64 weka fs tier fetch"
            subprocess.run(cmd, shell=True, check=True, text=True)
            print(f"Finished warming up {folder_path}")

        except subprocess.CalledProcessError as e:
            print(f"Error warming up {folder_path}: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Option 2: With a YAML-style string
    yaml_input = """
      - /fsx/loubna/datasets/llama_tokenized/fineweb-edu/merged
      - /fsx/loubna/datasets/llama_tokenized/other_sources/dclm/
      - /fsx/loubna/datasets/llama_tokenized/pes2o/standard
      - /fsx/loubna/datasets/llama_tokenized/other_sources/wiki
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-fra_Latn/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-spa_Latn/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-deu_Latn/
      - /fsx/loubna/datasets/llama_tokenized/fw2-hq-ita_Latn/standard
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-por_Latn/
      - /fsx/loubna/datasets/llama_tokenized/fw2-hq-cmn_Hani/standard
      - /fsx/loubna/datasets/llama_tokenized/fw2-hq-rus_Cyrl/standard
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-fas_Arab/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-jpn_Jpan/
      - /fsx/loubna/datasets/llama_tokenized/fw2-kor_Hang/standard
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hin_Deva/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-tha_Thai/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-vie_Latn/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/fw2-hq-ell_Grek/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/infiwebmath-3plus/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/finemath-3plus/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/stack-edu-Python/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/stack-edu-Java/
      - /fsx/loubna/datasets/llama_tokenized/other_sources/stack-edu-JavaScript/
      - /fsx/loubna/datasets/llama_tokenized/kaggle/standard
    """

    # First warm up the datasets
    print("Starting dataset warm-up...")
    warmup_datasets(yaml_input)

    # Then check warm-up status
    print("\nChecking warm-up status...")
    warmup_stats = get_warmup_percentages(yaml_input)
