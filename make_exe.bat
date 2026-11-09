@echo off
:: Usage: make_exe.bat [cpu|openvino|dml|xpu]   (default: openvino)
:: Builds dist\facetracker with the chosen inference backend bundled.
setlocal
set FLAVOR=%1
if "%FLAVOR%"=="" set FLAVOR=openvino
set ROOT=%~dp0

where uv >nul 2>nul || (echo Installing uv & pip install uv)

echo Installing dependencies (%FLAVOR%)
if "%FLAVOR%"=="xpu" (
    uv sync --locked --extra openvino --extra xpu --extra mediapipe --group build || exit /b 1
) else (
    uv sync --locked --extra %FLAVOR% --extra mediapipe --group build || exit /b 1
)

echo Fetching MediaPipe models
uv run --no-sync python tools\fetch_mediapipe_models.py || exit /b 1
echo Converting MediaPipe models to ONNX for the GPU backends (uses a temporary TensorFlow env)
uv run --no-sync python tools\convert_mediapipe_models.py || exit /b 1

echo Generating portable models
uv run --no-sync --with onnx python tools\portable_models.py --no-verify || exit /b 1

set EXTRA=--collect-all onnxruntime --collect-all mediapipe
if "%FLAVOR%"=="openvino" set EXTRA=%EXTRA% --collect-all openvino
if "%FLAVOR%"=="xpu" set EXTRA=%EXTRA% --collect-all openvino --collect-all onnx2torch --collect-all torch

echo Running pyinstaller
uv run --no-sync pyinstaller facetracker.py --clean --noconfirm --specpath build --onedir %EXTRA% --add-binary "%ROOT%dshowcapture\*.dll;." --add-binary "%ROOT%escapi\*.dll;." --add-binary "%ROOT%run.bat;." || exit /b 1

echo Copying models
xcopy /e /i /y models dist\facetracker\models >nul

echo Deleting opencv videoio dlls
del /f /q dist\facetracker\_internal\cv2\opencv_videoio_* 2>nul

echo Finished: dist\facetracker
