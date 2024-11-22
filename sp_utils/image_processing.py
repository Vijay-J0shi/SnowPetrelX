from rich import print

import pandas as pd

file = "utils/dataset.csv"

# SNOW PETREL FEATURE
headers = ["behavior", "image_id", "image_file", "head_x", "head_y", "beak_base_x", "beak_base_y", 
                  "beak_tip_x", "beak_tip_y", "neck_x", "neck_y", "body1_x", "body1_y", 
                  "body2_x", "body2_y", "tail_base_x", "tail_base_y"]

SKIP_ROWS = 3
SKIP_COLS = 0
usecols = range(SKIP_COLS, len(headers))

dataset = pd.read_csv(file, skiprows=SKIP_ROWS, usecols=usecols, names=headers)

print(dataset.head(2).to_dict())
