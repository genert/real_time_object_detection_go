# Real Time Object Detection with OpenCV, Go, and Yolo v4

## Install dependencies

### MacOS

```bash
brew install opencv cmake ffmpeg@4
```

### Linux

For installing OpenCV, I suggest to follow https://gocv.io/getting-started/linux/ tutorial; it has build commands for building OpenCV with CUDA support as well.


## Download Yolo model

```bash
cd ./cmd/ml && ./download_data_v4.sh
```

## Run the Go program

```bash
cd ./cmd/ml && go build .
./ml --settings=config.json
```

## Use webcam for object detection

Change "source" in config.json to "webcam". Don't forget to check "device_id" value in "video_capture_device" object.


