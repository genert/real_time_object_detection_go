package ml

import (
	"fmt"
	"image"

	"github.com/pkg/errors"
	"gocv.io/x/gocv"
)

const (
	yoloScaleFactor = 1.0 / 255.0
	yoloHeight      = 608
	yoloWidth       = 608
)

var (
	yoloSize     = image.Point{X: yoloHeight, Y: yoloWidth}
	yoloMean     = gocv.NewScalar(0.0, 0.0, 0.0, 0.0)
	yoloBlobName = ""
)

// DetectedObject Store detected object info
type DetectedObject struct {
	// Bounding box
	Rect image.Rectangle
	// Class identifier
	ClassID int
	// Class description (in most of cases this is just another class unique identifier)
	ClassName string
	// The probability that an object belongs to the specified class
	Confidence float32

	// Unexported
	speed float32
}

// String returns something we call 'hash' for detected object
func (d *DetectedObject) String() string {
	return fmt.Sprintf("DetectedObject{classID: %d, conf: %.5f, rect: ((%d, %d), (%d, %d))}", d.ClassID, d.Confidence, d.Rect.Min.X, d.Rect.Min.Y, d.Rect.Max.X, d.Rect.Max.Y)
}

// DetectObjects Detect objects for provided Go's image via neural network
//
// app - Application instance containing pointer to neural network for object detection
// img - gocv.Mat image object
// netClasses - neural network predefined classes
// filters - List of classes for which you need to filter detected objects
//
func DetectObjects(app *Application, img gocv.Mat, netClasses []string, filters ...string) ([]*DetectedObject, error) {
	blobImg := gocv.BlobFromImage(img, yoloScaleFactor, yoloSize, yoloMean, true, false)
	defer blobImg.Close()

	app.neuralNetwork.SetInput(blobImg, yoloBlobName)
	detections := app.neuralNetwork.ForwardLayers(app.layersNames)
	detected, err := postprocess(detections, 0.5, 0.4, float32(img.Cols()), float32(img.Rows()), netClasses, filters)

	for i := range detections {
		err := detections[i].Close()
		if err != nil {
			return detected, errors.Wrap(err, "Can't deallocate gocv.Mat")
		}
	}
	return detected, err
}

func postprocess(detections []gocv.Mat, confidenceThreshold, nmsThreshold float32, frameWidth, frameHeight float32, netClasses []string, filters []string) ([]*DetectedObject, error) {
	var detectedObjects []*DetectedObject
	var bboxes []image.Rectangle
	var confidences []float32

	for i, yoloLayer := range detections {
		cols := yoloLayer.Cols()
		data, err := detections[i].DataPtrFloat32()
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract data")
		}
		for j := 0; j < yoloLayer.Total(); j += cols {
			row := data[j : j+cols]
			scores := row[5:]
			classID, confidence := getClassIDAndConfidence(scores)
			className := netClasses[classID]
			if stringInSlice(&className, filters) {
				if confidence > confidenceThreshold {
					confidences = append(confidences, confidence)
					boundingBox := calculateBoundingBox(frameWidth, frameHeight, row)
					bboxes = append(bboxes, boundingBox)
					detectedObjects = append(detectedObjects, &DetectedObject{
						Rect:       boundingBox,
						ClassName:  className,
						ClassID:    classID,
						Confidence: confidence,
					})
				}
			}
		}
	}
	if len(bboxes) == 0 {
		return nil, nil
	}
	indices := make([]int, len(bboxes))
	for i := range indices {
		indices[i] = -1
	}
	gocv.NMSBoxes(bboxes, confidences, confidenceThreshold, nmsThreshold, indices)
	filteredDetectedObjects := make([]*DetectedObject, 0, len(detectedObjects))
	for i, idx := range indices {
		if idx < 0 || (i != 0 && idx == 0) {
			// Eliminate zeros, since they are filtered by NMS (except first element)
			// Also filter all '-1' which are undefined by default
			continue
		}
		filteredDetectedObjects = append(filteredDetectedObjects, detectedObjects[idx])
	}
	return filteredDetectedObjects, nil
}

func getClassIDAndConfidence(x []float32) (int, float32) {
	res := 0
	max := float32(0.0)
	for i, y := range x {
		if y > max {
			max = y
			res = i
		}
	}
	return res, max
}

func calculateBoundingBox(frameWidth, frameHeight float32, row []float32) image.Rectangle {
	if len(row) < 4 {
		return image.Rect(0, 0, 0, 0)
	}
	centerX := int(row[0] * frameWidth)
	centerY := int(row[1] * frameHeight)
	width := int(row[2] * frameWidth)
	height := int(row[3] * frameHeight)
	left := centerX - width/2
	top := centerY - height/2
	return image.Rect(left, top, left+width, top+height)
}

func stringInSlice(str *string, sl []string) bool {
	for i := range sl {
		if sl[i] == *str {
			return true
		}
	}
	return false
}
