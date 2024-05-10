import subprocess
from pathlib import Path

with Path("urls").open() as f:
    urls = [line.strip() for line in f.readlines()]

for url in urls[4:6]:
    subprocess.run(["python", "synthetic_data.py", url], check=True)
