"""Merge the four systems' results into one table, ordered by storage.

Refuses to merge results whose provenance differs. The two CSVs are produced by separate
commands, so nothing stops one from being stale — and a merged table that silently mixes
turbovec measured on 25,000 vectors with zquant measured on 100,000 looks entirely
plausible. That failure mode is not hypothetical: a reported 1.15x gap was really 1.26x
because the two sides were measured at different retrieval depths (docs/notes.md).
"""
import csv, sys

PROVENANCE = ("dataset", "n", "d", "nq", "k")
# `threads` is recorded but deliberately not part of the match key: the baselines manage
# their own pool and report 0, while zquant names its thread count. Both columns are
# full-machine throughput, which is what makes them comparable.

rows = []
seen = {}
for path in ("data/baselines.csv", "data/zquant.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        if not set(PROVENANCE) <= set(reader.fieldnames or ()):
            sys.exit(f"{path} predates provenance stamping - regenerate it before merging")
        for r in reader:
            key = tuple(r[c] for c in PROVENANCE)
            seen.setdefault(key, []).append(path)
            r["bytes_per_vector"] = float(r["bytes_per_vector"])
            r["recall_at_10"] = float(r["recall_at_10"])
            r["rms"] = float(r.get("rms") or 0) or None
            rows.append(r)

if len(seen) > 1:
    print("refusing to merge: these results were not measured under the same conditions\n")
    for key, paths in seen.items():
        where = ", ".join(sorted(set(paths)))
        print("  " + "  ".join(f"{c}={v}" for c, v in zip(PROVENANCE, key)) + f"   [{where}]")
    sys.exit("\nre-run whichever side is stale, then merge")

cond = next(iter(seen))
print("  ".join(f"{c}={v}" for c, v in zip(PROVENANCE, cond)))
print()

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
# Bands span the whole measured range; an earlier list stopped at 90 B and hid the
# budgets where zquant leads.
bands = [(0, 25), (25, 40), (40, 56), (56, 72), (72, 90),
         (90, 110), (110, 140), (140, 200)]
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
