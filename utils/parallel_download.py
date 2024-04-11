import time

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from kinetics_downloader import download_kinetics_class


def download_kinetics_class_multithreaded(class_data: pd.DataFrame, download_path: str, log_path: str = None,
                                          num_threads: int = 1) -> pd.DataFrame:
    # Split the DataFrame into num_threads parts
    data_splits = np.array_split(class_data, num_threads)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for data in data_splits:
            # Schedule the download_kinetics_class function to be executed and return a Future object
            futures.append(executor.submit(download_kinetics_class, data, download_path, log_path))

        results = []
        for future in futures:
            # Wait for the function to complete and get the result
            results.append(future.result())

    # Concatenate the results into a single DataFrame
    return pd.concat(results)


class_data = pd.read_csv("../data/kinetics_700/dancing.csv")
# count time
start_time = time.time()
download_kinetics_class_multithreaded(class_data, "../data/kinetics_700/videos", "log.txt", num_threads=8)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds") #160
