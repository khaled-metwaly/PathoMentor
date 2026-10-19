@echo off
REM Train both arms x seeds 0-4 at one FOV, one process per run, then run the statistics.
REM usage: scripts\run_all_seeds.bat <fov> <data_root> <teacher_cache.npy> [out_dir]
setlocal
set FOV=%~1
set DATA=%~2
set TEACHER=%~3
set OUT=%~4
if "%OUT%"=="" set OUT=runs
if "%TEACHER%"=="" (echo usage: run_all_seeds.bat ^<fov^> ^<data_root^> ^<teacher_cache.npy^> [out_dir] & exit /b 1)
for %%S in (0 1 2 3 4) do (
  if not exist "%OUT%\baseline_fov%FOV%_seed%%S\baseline_fov%FOV%_seed%%S_results.json" python -m pathomentor.train --arm baseline --fov %FOV% --seed %%S --data-root "%DATA%" --out-dir "%OUT%" || exit /b 1
  if not exist "%OUT%\mentor_fov%FOV%_seed%%S\mentor_fov%FOV%_seed%%S_results.json" python -m pathomentor.train --arm mentor --fov %FOV% --seed %%S --data-root "%DATA%" --teacher-cache "%TEACHER%" --out-dir "%OUT%" || exit /b 1
)
python -m pathomentor.compare --runs "%OUT%" --out-dir "%OUT%\compare"
endlocal
