package ml

import (
	"image"

	"gocv.io/x/gocv"
)

// FrameData Wrapper around gocv.Mat
type FrameData struct {
	ImgSource     gocv.Mat //  Source image
	ImgScaled     gocv.Mat // Scaled image
	ImgScaledCopy gocv.Mat // Copy of scaled image
}

// NewFrameData Simplifies creation of FrameData
func NewFrameData() *FrameData {
	fd := FrameData{
		ImgSource: gocv.NewMat(),
		ImgScaled: gocv.NewMat(),
	}
	return &fd
}

// Close Simplify memory management for each gocv.Mat of FrameData
func (fd *FrameData) Close() {
	_ = fd.ImgSource.Close()
	_ = fd.ImgScaled.Close()
	_ = fd.ImgScaledCopy.Close()
}

// Preprocess Scales image to given width and height
func (fd *FrameData) Preprocess(width, height int) error {
	gocv.Resize(fd.ImgSource, &fd.ImgScaled, image.Point{X: width, Y: height}, 0, 0, gocv.InterpolationDefault)
	fd.ImgScaledCopy = fd.ImgScaled.Clone()
	return nil
}
