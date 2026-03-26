import subprocess
from pathlib import Path

VISION_GUI = Path(r"D:\work\corona\Vision\cmake-build-release\bin\vision-gui.exe")

for d in sorted(Path(__file__).parent.iterdir()):
    if d.is_dir():
        p = d / "vision_scene.json"
        if p.exists():
            print(f"Rendering {p.resolve()} ...")
            subprocess.run([
                str(VISION_GUI),
                "-s", str(p.resolve()),
                "-n", "20",
                "-o", "golden_image.exr",
            ])
