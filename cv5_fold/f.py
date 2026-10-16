import pandas as pd
d = pd.read_csv("cv5_fold/cv5_drift_per_stream.csv")
keys = list(d.columns[:list(d.columns).index("policy")])
p = d.pivot_table(index=keys, columns="policy", values=["retrains", "post_drift_acc"])
print("cost_gate == every_flag retrains:", (p["retrains"]["cost_gate"] == p["retrains"]["every_flag"]).mean())
print(p["post_drift_acc"].mean().round(4))
print(p["retrains"].mean().round(2))
