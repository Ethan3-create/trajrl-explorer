import zipfile
import os
import sys

# Source files
base = r"C:\Users\PRLOUIS\Desktop\项目与代码\trajrl_paper_repro_workdir"
files = [
    "web_server.py",
    "validate.py",
    "web_demo/index.html",
    "web_demo/css/style.css",
    "web_demo/js/app.js",
    "web_demo/README.md",
]

out = os.path.join(base, "TrajRL_Web_Demo.zip")
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        src = os.path.join(base, f)
        if os.path.exists(src):
            zf.write(src, f)
            print(f"  + {f}")
        else:
            print(f"  - {f} NOT FOUND")

size = os.path.getsize(out)
print(f"\nCreated: {out}")
print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")
