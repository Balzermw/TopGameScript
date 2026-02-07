## ✅ What this script does

✅ Dry-run never extracts (no long 7z operations)

✅ Real run does **NOT** extract the whole archive — it:
- lists contents (`7z l -slt`)
- extracts only the relevant files for that system (e.g., GameCube: `.rvz`, `.gcz`, `.iso`)

✅ Skips junk “systems” like `.top_games_cache` and `Top Games`

✅ Still overwrites the output folder each run

✅ Still flat output (one folder per system, no per-game subfolders)

✅ Adds optional `--extract-timeout` so a single archive can’t hang your whole run

## Run (recommended)

### Fast test (no Reddit, no extraction)

```powershell
py top_games_flat.py --library-root "\\The-Epoch\DriveMan\Roms" --dry-run --no-reddit
```

### Real run

```powershell
py top_games_flat.py --library-root "\\The-Epoch\DriveMan\Roms"
```

### If 7z isn’t on PATH

```powershell
py top_games_flat.py --library-root "\\The-Epoch\DriveMan\Roms" --sevenzip "C:\Program Files\7-Zip\7z.exe"
```

### If you want to prevent “stuck” extraction

```powershell
py top_games_flat.py --library-root "\\The-Epoch\DriveMan\Roms" --extract-timeout 600
```
