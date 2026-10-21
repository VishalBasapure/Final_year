from pathlib import Path
import pandas as pd
import numpy as np

raw_dir = Path(__file__).resolve().parent / "data" / "raw"

for f in sorted(raw_dir.glob("*.csv")):
    a = pd.read_csv(f)[["ax_g", "ay_g", "az_g"]].values
    print(f, round(float(np.linalg.norm(a, axis=1).mean()), 3))