import json

from config import dataset_path
from profiler import load_dataset
from detector import run_quality_report


path = dataset_path()
df = load_dataset(path)

report = run_quality_report(df)

summary = report["dataset_summary"]
print(f"=== {path} : UNIFIED QUALITY REPORT ===")
print(f"{summary['total_rows']:,} rows x {summary['total_columns']} columns")
print(f"column types: {report['checks']['column_types']['summary']['type_counts']}")

print(f"\nissues found ({report['total_issue_types_found']} of {len(report['issues_found'])}):")
for name, found in report["issues_found"].items():
    print(f"  [{'x' if found else ' '}] {name}")

# Per-column detail is large on a wide file, so write it out instead of
# flooding the terminal. default=str keeps it safe for any dtype.
with open("report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)
print("\nfull detail -> report.json")