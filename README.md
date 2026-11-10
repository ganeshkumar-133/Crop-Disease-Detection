# CropShield AI — Improved ONNX Plant Disease Detection

## What changed
- Image color detection now measures the percentage of chromatic pixels, so a mostly white powdery leaf with visible green areas is less likely to be mislabeled as black-and-white.
- Displays the percentage of colored pixels and warns about low-resolution images.
- Shows a readable best-match class name and the top three model scores.
- Labels the number as a **top-class model score**, not guaranteed diagnostic confidence. Warns when the top result is weak or close to the runner-up.
- More robust parsing of supported `classes.json` formats and clearer model setup errors.

## Required model files
Download these two files from the same model repository: https://huggingface.co/AbhiCommits/cropguard-models/tree/main
- `cropguard.onnx`
- `classes.json`

Copy both into `Crop-Disease-Detection-Recreated\models\`. Keep their filenames unchanged. The model is licensed CC BY-SA 4.0; keep attribution and share-alike terms when redistributing adapted versions.

## Exact Windows setup
1. Stop the currently running app: focus the black Command Prompt window opened by `run.bat`, press **Ctrl+C**, and wait for it to stop.
2. Extract this improved ZIP into a new folder (for example, Downloads). Avoid extracting over the old folder.
3. Open `Crop-Disease-Detection-Recreated`, then open its `models` folder. Copy `cropguard.onnx` and `classes.json` into it. Confirm the files are named exactly as shown.
4. Double-click `setup.bat`. Wait until it says setup is complete. It needs internet to install Python packages.
5. Double-click `run.bat`. Keep its black window open.
6. Open `http://localhost:8501` in your browser and upload a clear, close-up leaf image.
7. Check the color classification, image resolution note, best-match label, top-class model score, and top-three scores.

If setup fails, copy the full error from the black window. Do not close it before copying the message.

## Requirements
- Windows and Python 3.10 available as `py -3.10`
- CPU only; no Raspberry Pi and no paid API

## Limitations
The CropGuard model is trained on PlantVillage close-up detached-leaf images. It lists 38 classes across 14 crops but does not cover every crop or disease. Whole-field drone images are not validated. The model always selects from its supported classes; a high model score is not proof that a crop/disease is supported or correctly diagnosed. Use as a student project aid, not as definitive agricultural advice.
