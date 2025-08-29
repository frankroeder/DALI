import pandas as pd
import pickle
import os

def get_df_from_file(filename):
    print(f"Processing {filename}")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    rows = []
    for entry in data:
        context_dict = entry["context"]
        context_order = entry["context_order"]
        # Build an ordered tuple of the context values. For example: (4.9, 1.0)
        context_tuple = tuple(context_dict[k] for k in context_order)

        # Each entry has one or more episodes (each with several samples)
        episodes = entry["episodes"]
        for ep_idx, ep in enumerate(episodes):
            for sample_idx in range(ep["obs"].shape[0]):
                #print(ep["posterior"].shape)
                # TODO: check if "imagined" is shorter
                row = {}
                for key in ep.keys():
                    if ep[key].shape[0] != ep["obs"].shape[0]:
                        # For debugging
                        #print(f"Skipping {key} {ep[key].shape=}")
                        continue
                    row[key] = ep[key][sample_idx]
                row["episode"] = ep_idx
                row["real_context"] = context_tuple
                row["sample"] = sample_idx
                if "embed" in ep.keys():
                    row["embed"] = ep["embed"][sample_idx]
                rows.append(row)

    return pd.DataFrame(rows)

def create_folder(folder_path):
    # Check if folder exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")