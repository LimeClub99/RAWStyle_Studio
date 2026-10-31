# RAWStyle Studio

RAWStyle Studio is a local app that trains a model on your RAW photos and Lightroom Classic edits, then predicts adjustments for new photos. A browser interface handles training and prediction, and a Lightroom plug-in applies the results.

## Requirements

- macOS with Python 3.10 or newer.
- Lightroom Classic.
- ExifTool installed and available as `exiftool`.

## Setup

1. Download and extract the project.
2. Open `Start RAWStyle Studio.command`. The first run installs the Python dependencies and opens the app in your browser.
3. In Lightroom Classic, open **File → Plug-in Manager → Add** and select the included `RAWStyle Lightroom.lrplugin` folder.

## Usage

1. Select your Lightroom Classic `.lrcat` file and the folder containing its original RAW photos.
2. Click **Train model**. At least 20 matching RAW/edit pairs are required.
3. Select a folder of new RAW photos and click **Predict adjustments**.
4. Click **Show predictions in Finder** to locate the generated file.
5. In Lightroom Classic, import and select the new photos. Choose **Library → Plug-in Extras → Apply RAWStyle Predictions…**, select the predictions file, and click **Apply**.

Keep the Terminal window open while using the app. Press **Ctrl+C** in Terminal to stop it.

