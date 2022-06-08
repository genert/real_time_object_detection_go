package ml

import (
	"encoding/json"
	"fmt"
	"github.com/pkg/errors"
	"io/ioutil"
	"os"
	"strings"
	"sync"
)

// AppSettings Settings for application
type AppSettings struct {
	Source                     string                      `json:"source"`
	NeuralNetworkSettings      NeuralNetworkSettings       `json:"neural_network_settings"`
	CameraSettings             *CameraSettings             `json:"camera_settings"`
	VideoCaptureDeviceSettings *VideoCaptureDeviceSettings `json:"video_capture_device"`
	VideoSettings              *VideoSettings              `json:"video_settings"`
	MjpegSettings              MjpegSettings               `json:"mjpeg_settings"`

	sync.RWMutex
}

// NewSettings Create new AppSettings from content of configuration file
func NewSettings(fileName string) (*AppSettings, error) {
	jsonFile, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer jsonFile.Close()

	bytesValues, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		return nil, err
	}

	settings := AppSettings{}
	if err = json.Unmarshal(bytesValues, &settings); err != nil {
		return nil, err
	}

	if settings.Source == "" {
		return nil, fmt.Errorf("source setting is empty")
	}

	// Prepare video settings
	if settings.VideoSettings == nil {
		return nil, fmt.Errorf("field 'video_settings' has not been provided in configuration file")
	}
	settings.VideoSettings.Prepare()

	// Prepare Darknet's classes
	content, err := ioutil.ReadFile(settings.NeuralNetworkSettings.DarknetClasses)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read Darknet's classes file")
	}
	settings.NeuralNetworkSettings.NetClasses = strings.Split(string(content), "\n")

	return &settings, nil
}

// MjpegSettings settings for output
type MjpegSettings struct {
	ImshowEnable bool `json:"imshow_enable"`
	Enable       bool `json:"enable"`
	Port         int  `json:"port"`
}

// CameraSettings settings for camera settings
type CameraSettings struct {
	Address string `json:"address"`
	Port    int    `json:"port"`
	Width   int    `json:"width"`
	Height  int    `json:"height"`
}

// VideoCaptureDeviceSettings settings for device settings
type VideoCaptureDeviceSettings struct {
	DeviceID int `json:"device_id"`
}

// NeuralNetworkSettings Neural network
type NeuralNetworkSettings struct {
	Enable         bool    `json:"enable"`
	Target         string  `json:"target"`
	Backend        string  `json:"backend"`
	DarknetCFG     string  `json:"darknet_cfg"`
	DarknetWeights string  `json:"darknet_weights"`
	DarknetClasses string  `json:"darknet_classes"`
	ConfThreshold  float64 `json:"conf_threshold"`
	NmsThreshold   float64 `json:"nms_threshold"`
	// Exported, but not from JSON
	NetClasses    []string `json:"-"`
	TargetClasses []string `json:"target_classes"`
}
