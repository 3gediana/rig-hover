from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"D:\robot_work\evidence\sessions")
OUT = Path(r"D:\robot_work\stream_memory_research\agent_feature_probe\contact_sheets")
OUT.mkdir(parents=True, exist_ok=True)
SESSIONS = {
    "knock": "knock_d0.5_front_night_182129",
    "wave": "wave_d0.5_front_night_182035",
    "leave_return": "lean_out_d0.5_front_night_182257",
}

for name, folder in SESSIONS.items():
    files = sorted((ROOT / folder / "frames").glob("*.jpg"))
    page_frames = 50
    for page, begin in enumerate(range(0, len(files), page_frames)):
        ids = list(range(begin, min(begin + page_frames, len(files))))
        tiles = []
        for i in ids:
            im = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.resize(im, (120, 120), interpolation=cv2.INTER_AREA)
            tile = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(tile, (0, 0), (119, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"{i:03d}  t={i/10:.1f}s", (3, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        cols = 5
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * 124, cols * 124, 3), 245, np.uint8)
        for j, tile in enumerate(tiles):
            y, x = (j // cols) * 124, (j % cols) * 124
            sheet[y:y+120, x:x+120] = tile
        path = OUT / f"{name}_{page:02d}.jpg"
        cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(name, page, len(ids), path)
