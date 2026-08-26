"""Merge the four systems' results into one table, ordered by storage."""
import csv

rows = []
for path in ("data/baselines.csv", "data/zquant.csv"):
    with open(path) as f:
        for r in csv.DictReader(f):
            r["bytes_per_vector"] = float(r["bytes_per_vector"])
            r["recall_at_10"] = float(r["recall_at_10"])
            r["rms"] = float(r.get("rms") or 0) or None
            rows.append(r)

rows.sort(key=lambda r: (r["bytes_per_vector"], -r["recall_at_10"]))

hdr = f'{"system":<14}{"config":<22}{"B/vec":>7}{"R@10":>8}{"med":>5}{"p90":>5}{"worst":>7}{"rms":>10}{"QPS":>10}'
print(hdr)
print("-" * len(hdr))
for r in rows:
    rms = f'{r["rms"]:.5f}' if r["rms"] else "-"
    print(f'{r["system"]:<14}{r["config"]:<22}{r["bytes_per_vector"]:>7.1f}'
          f'{r["recall_at_10"]:>8.3f}{r["rank_median"]:>5}{r["rank_p90"]:>5}'
          f'{r["rank_worst"]:>7}{rms:>10}{float(r["qps"]):>10.0f}')

print()
print("Best R@10 within each storage band:")
bands = [(0, 25), (25, 40), (40, 56), (56, 72), (72, 90)]
for lo, hi in bands:
    inb = [r for r in rows if lo <= r["bytes_per_vector"] < hi]
    if not inb:
        continue
    best = max(inb, key=lambda r: r["recall_at_10"])
    ours = [r for r in inb if r["system"] == "zquant"]
    ours_best = max(ours, key=lambda r: r["recall_at_10"]) if ours else None
    line = f'  {lo:>3}-{hi:<3} B: {best["system"]} {best["config"]} = {best["recall_at_10"]:.3f}'
    if ours_best and ours_best is not best:
        deficit = best["recall_at_10"] - ours_best["recall_at_10"]
        line += f'   (zquant best {ours_best["recall_at_10"]:.3f}, -{deficit:.3f})'
    print(line)
