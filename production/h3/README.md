# H3 Production Pipeline (MiniMax H3 on WanGP, 3090)

Canonical copy of the music-video / audio-conditioned generation pipeline.
Mirrored from the 3090 (`straughter@3090:/home/straughter/h3_runs/`) on 2026-08-15.
The 3090 copy remains the live working set — this directory is the durable backup + reviewable source of truth.

## Layout

| File | Purpose |
|---|---|
| `beat_grid_cut.py` | Snap multishot cuts to a track's beat grid, reassemble, mux soundtrack. Two modes: bed (dialogue+ducked loop) and MUSIC_ONLY (full song as soundtrack). |
| `multitake.sh` | Run one job JSON across N seeds, rename outputs for QC-pick. `DRAFT_STEPS=8` for fast iteration. |
| `*.json` | Proven job configs (see below). |
| `jobs/` | Historical/superseded job configs. |

## Where things live on the 3090

- Job configs, guides, stills: `/home/straughter/h3_runs/`
- Generated video: `/home/straughter/Wan2GP/outputs/`
- Keeper songs (mastered bangers): `/mnt/bulk/home/straughter/sgflix_audio_factory/keepers/`
- WanGP app + venv: `/home/straughter/Wan2GP/` (`source venv/bin/activate`)

## Running a job

```bash
# always in tmux (long renders); GPU must be idle
tmux new-session -d -s run "cd /home/straughter/Wan2GP && source venv/bin/activate && \
  export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 && \
  python3 wgp.py --process /home/straughter/h3_runs/<job>.json --profile 3 --attention sdpa --verbose 1 \
  2>&1 | tee /tmp/<job>.log"
```

## Key model types

- `minimax_h3_fl2va_pruned` — text/image→video, multishot via `script` (shots separated by `\n---\n`).
- `minimax_h3_ref2va_pruned` — reference-image + **audio_guide** → video with generated audio. **This is the native lip-sync path.**

## Native lip-sync recipe (verified 2026-08-15, Wet Reckless)

Ref2VA does **not** copy the reference audio. It generates a new vocal/track rendering *guided* by it —
same melody contour, tempo, phrasing, language — and the generated audio drives the mouth motion.
The generated audio is discarded in post; the keeper track is muxed back as the soundtrack.

1. **Measure where the final video sits on the track — never assume.**
   Spectral-cosine match of the cut's audio against the keeper locates the base offset
   (raw waveform correlation is unreliable after post-processing ducking/EQ; spectral cosine ~0.99 = same recording).
2. Cut each shot's `audio_guide` slice from the keeper with ffmpeg (`-ss <t> -t <len> -c:a pcm_s16le`).
   Guide length must match `video_length`/24fps exactly.
3. Identity still: pull a frontal, open-mouth, eyes-open frame from the existing render (f-score via vision check ≥8.5).
4. Ref2VA job: `video_prompt_type: "I"`, `image_refs: [<still>]`, `audio_guide: <slice>`,
   480×832, 20 steps, `guidance_scale: 1.0`, `embedded_guidance_scale: 6.0`, `force_fps: "24"`.
5. QC with ModelScope Qwen video analysis (re-encode to 320px first to avoid 504s).
6. Mux: `MUSIC_ONLY=1 MUSIC_VOL=1.0 MUSIC_START=<base> python3 beat_grid_cut.py <keeper.wav> <video.mp4> <out.mp4> <cut_frames...>`

## Proven configs

| JSON | What it demonstrates |
|---|---|
| `wet_reckless_mv_final.json` | 3-shot fl2va multishot, cuts on beats 16/32, 143.6 BPM hook |
| `wr_ref2va_shot1_v2.json` / `wr_ref2va_shot2_v2.json` | ref2va lip-sync regen with **measured** keeper windows (16.90–22.23s / 22.19–29.53s) |
| `wr_ref2va_shot2.json` | first ref2va lip-sync test (wrong window 7.18s — kept for reference) |
| `audio_master_pacing_v1.json` | ref2va pacing from audio_master |
| `unc_ray_job_hoa_v1.json` | Uncle Ray dialogue scene (character lock from `data/character_locks.json`) |

## Gotchas (learned the hard way)

- Anchor math: video t=0 maps to `anchor` = first detected beat ≥ MUSIC_START, NOT to keeper t=0 or first beat of file.
- Scene-detect misses soft seams — use frame-diff curves to find real cut points.
- Raw-waveform correlation fails on post-processed audio; use onset-envelope or spectral cosine.
- Generated audio ≠ keeper: never expect the input track in the render output.
