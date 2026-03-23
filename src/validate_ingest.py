"""Quick validation script: compare ingested parquets against stats.txt ground truth."""

import pandas as pd

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, got, expected):
    ok = got == expected
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}: got {got:,}  (expected {expected:,})")
    return ok

print("Loading parquets...")
pl = pd.read_parquet("processed/playlists.parquet")
tr = pd.read_parquet("processed/tracks.parquet")
tm = pd.read_parquet("processed/track_meta.parquet")

print("\n=== Core counts ===")
results = []
results.append(check("num playlists",        len(pl),                     1_000_000))
results.append(check("num track entries",    len(tr),                    66_346_428))
results.append(check("num unique tracks",    tr["track_uri"].nunique(),   2_262_292))
results.append(check("num unique albums",    tr["album_uri"].nunique(),     734_684))
results.append(check("num unique artists",   tr["artist_uri"].nunique(),    295_860))
results.append(check("num unique titles",    pl["name"].nunique(),           92_944))
results.append(check("num playlists w/ desc",pl["has_desc"].sum(),           18_760))

avg_len = round(len(tr) / len(pl), 6)
expected_avg = 66.346428
ok = abs(avg_len - expected_avg) < 0.001
print(f"  [{'PASS' if ok else 'FAIL'}] avg playlist length: got {avg_len:.6f}  (expected {expected_avg})")
results.append(ok)

print("\n=== track_meta sanity ===")
results.append(check("track_meta rows == unique tracks", len(tm), 2_262_292))

print("\n=== Top track spot-check ===")
# stats.txt counts per unique track_uri; find the most-played URI named "HUMBLE."
track_counts = tr.groupby("track_uri").size()
humble_uris = tr[tr["track_name"] == "HUMBLE."]["track_uri"].unique()
if len(humble_uris) > 0:
    humble_max = track_counts[humble_uris].max()
    ok = humble_max == 46_574
    print(f"  [{'PASS' if ok else 'FAIL'}] HUMBLE. top URI appearances: got {humble_max:,}  (expected 46,574)")
    if len(humble_uris) > 1:
        print(f"    (note: {len(humble_uris)} distinct URIs share this title — stats.txt counts the top one)")
    results.append(ok)
else:
    print("  [FAIL] HUMBLE. not found in tracks")
    results.append(False)

print("\n=== Top artist spot-check ===")
drake_count = tr[tr["artist_name"] == "Drake"].shape[0]
ok = drake_count == 847_160
print(f"  [{'PASS' if ok else 'FAIL'}] Drake appearances: got {drake_count:,}  (expected 847,160)")
results.append(ok)

print("\n=== Summary ===")
passed = sum(results)
total = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  All checks PASSED — ingestion looks correct.")
else:
    print(f"  {total - passed} check(s) FAILED — investigate above.")