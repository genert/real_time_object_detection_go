
# Real Time Object Detection with OpenCV, Go, and Yolo v4

## Install dependencies

```bash
brew install opencv ffmpeg@4
```

## Run the Go program

```bash
cd ./cmd/ml && ./download_data_v4.sh
go build .
./ml --settings=config.json
```
