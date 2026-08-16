#!/usr/bin/env python3
"""Beat-grid cut tool: snap multishot cut points to a music bed's beat grid,
then reassemble the video with cuts on beats and mix the track underneath.

Two modes:

1) BED MODE (default) — short bed looped under generated dialogue:
   bed ducked to 0.55, generated audio kept, bed looped over its exact-beat
   window [t0, t_last] so the seam lands on a beat.

2) MUSIC VIDEO MODE (MUSIC_ONLY=1) — full song as the soundtrack:
   generated audio muted, song at full volume (MUSIC_VOL), trimmed straight
   from MUSIC_START (sec) — no loop needed when the song outlasts the video.
   The beat grid is re-phased to MUSIC_START so cuts still lock.

Env knobs: MUSIC_ONLY, MUSIC_VOL, MUSIC_START
Usage: python3 beat_grid_cut.py <music.wav> <video.mp4> <out.mp4> [cut_frames...]
"""
import json
import os
import subprocess
import sys

import librosa
import numpy as np


def ffprobe_dur(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", path]).decode().strip())


def main():
    bed_path, video_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    shot_frames = [int(x) for x in sys.argv[4:]] or [107, 214, 321]
    fps = 24.0
    snap_tol = 0.30
    music_only = os.environ.get("MUSIC_ONLY", "0") == "1"
    music_vol = float(os.environ.get("MUSIC_VOL", "0.55"))
    music_start = float(os.environ.get("MUSIC_START", "0"))

    dur = ffprobe_dur(video_path)

    # --- measure beat grid ---
    y, sr = librosa.load(bed_path, sr=22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    if music_only:
        # MUSIC VIDEO: re-phase grid to MUSIC_START. Anchor = first detected
        # beat at/after music_start; grid extends periodically from there.
        song_dur = len(y) / sr
        later = beat_times[beat_times >= music_start]
        if len(later) == 0:
            raise SystemExit("no detected beats at/after MUSIC_START")
        anchor = float(later[0])
        interval = float(np.median(np.diff(later))) if len(later) > 1 else 60.0 / bpm
        # Video-time grid: video t=0 == song anchor (a detected beat), so beats
        # fall at n*interval in VIDEO time. (Old bug: grid was built in
        # song-time starting at `anchor`, putting every grid point past the
        # video's end — snapping silently never fired.)
        grid = np.arange(0, dur + interval, interval)
        trim_start, trim_end = anchor, min(anchor + dur, song_dur)
        loop_info = "direct trim (no loop)"
    else:
        # BED MODE: anchor = first detected beat; extend grid periodically.
        t0 = float(beat_times[0])
        t_last = float(beat_times[-1])
        n_intervals = max(1, len(beat_times) - 1)
        loop_len = t_last - t0
        interval = loop_len / n_intervals  # exact period of the beat-locked loop
        grid = list(beat_times)
        t = t_last
        while t < dur + interval:
            t += interval
            grid.append(t)
        grid = np.array(grid)
        anchor, trim_start, trim_end = t0, t0, t_last
        loop_info = f"loop [{t0:.3f},{t_last:.3f}] ({loop_len:.3f}s)"

    # --- snap cuts to nearest grid point ---
    orig = [f / fps for f in shot_frames]
    snapped_sec, moved = [], []
    for b in orig:
        idx = int(np.argmin(np.abs(grid - b)))
        g = float(grid[idx])
        if abs(g - b) <= snap_tol:
            snapped_sec.append(g)
            moved.append(round(g - b, 3))
        else:
            snapped_sec.append(b)
            moved.append(0.0)

    report = {
        "mode": "music_video" if music_only else "bed",
        "bpm": round(bpm, 2),
        "interval": round(interval, 4),
        "video_duration": dur,
        "music_start": music_start,
        "grid_anchor": round(anchor, 3),
        "trim_window": [round(trim_start, 3), round(trim_end, 3)],
        "bed_handling": loop_info,
        "original_cuts_sec": [round(x, 3) for x in orig],
        "snapped_cuts_sec": [round(x, 3) for x in snapped_sec],
        "snap_deltas_sec": moved,
    }
    print(json.dumps(report, indent=1))

    # --- ffmpeg filtergraph ---
    bounds = [0.0] + snapped_sec + [dur]
    vparts, aparts = [], []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        fa, fb = int(round(a * fps)), int(round(b * fps))
        vparts.append(f"[0:v]trim=start_frame={fa}:end_frame={fb},setpts=PTS-STARTPTS[v{i}]")
        if not music_only:
            aparts.append(f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{i}]")
    n = len(bounds) - 1
    vconcat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]"

    if music_only:
        # song only: trim from anchor, volume, pad/trim to video length
        amix = (
            f"[1:a]atrim=start={trim_start:.4f}:end={trim_end:.4f},"
            f"asetpts=PTS-STARTPTS,volume={music_vol},"
            f"apad=whole_dur={dur:.4f},"
            f"atrim=end={dur:.4f}[aout]"
        )
        fc = ";".join(vparts + [vconcat, amix])
    else:
        aconcat = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[amix0]"
        bedmix = (
            f"[1:a]atrim=start={trim_start:.4f}:end={trim_end:.4f},"
            f"asetpts=PTS-STARTPTS,volume={music_vol}[music];"
            f"[amix0][music]amix=inputs=2:duration=first:dropout_transition=1[aout]"
        )
        fc = ";".join(vparts + aparts + [vconcat, aconcat, bedmix])

    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", bed_path,
           "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]
    print("\nRunning ffmpeg...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
