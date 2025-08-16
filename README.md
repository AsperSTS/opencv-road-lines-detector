# Video Processing for Road Line Highlighting

This project uses OpenCV to process videos and highlight road lines through image processing techniques. It allows real-time video visualization or exports the processed video as a file.

## Features

- **Real-time Processing**: View processed video with highlighted road lines in real-time
- **Video Export**: Save the processed video as an MP4 file
- **Road Line Detection**: Uses image processing algorithms to identify and highlight road markings
- **Flexible Input**: Support for various video formats

## Requirements

- Python 3.x
- OpenCV
- NumPy
- argparse

Install the required dependencies using pip:

```bash
pip install opencv-python numpy argparse
```

## Usage

To run the program, use the following command:

```bash
python main.py -m <mode> -v <video_path>
```

## Parameters

**-m or --modo: Defines the execution mode**

- `1`: View video in real-time
- `2`: Export video as MP4 file (default mode)

**-v or --video: Path to the video file to process (required parameter)**

## Examples

```bash
# Real-time visualization
python main.py -m 1 -v ./videos/highway.mp4

# Export processed video
python main.py -m 2 -v ./videos/highway.mp4

# Using default mode (export)
python main.py -v ./videos/highway.mp4
```

## How It Works

The program applies computer vision techniques to:

1. **Frame Extraction**: Reads video frames sequentially
2. **Preprocessing**: Applies filters and transformations to enhance line visibility
3. **Line Detection**: Uses edge detection and Hough transforms to identify road lines
4. **Highlighting**: Overlays detected lines on the original video frames
5. **Output**: Either displays in real-time or saves to file

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV
- And other formats supported by OpenCV

## Output

- **Real-time mode**: Opens a window showing the processed video with highlighted road lines
- **Export mode**: Saves the processed video as `output.mp4` in the project directory

## Technical Details

This project demonstrates:
- Computer vision applications in automotive technology
- Real-time video processing techniques
- Lane detection algorithms
- OpenCV library usage for practical applications

---

*Computer Vision Project - Road Line Detection System*