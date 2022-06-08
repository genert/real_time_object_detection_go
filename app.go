package ml

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/hybridgroup/mjpeg"
	"github.com/mike1808/h264decoder/decoder"
	"github.com/pkg/errors"
	"github.com/projecthunt/reuseable"
	"github.com/rs/cors"
	"gocv.io/x/gocv"
)

var colors = []color.RGBA{
	{R: 255, G: 255, B: 0},
	{R: 0, G: 255, B: 0},
	{R: 0, G: 255, B: 255},
	{R: 255, G: 0, B: 0},
}

// Application Main engine
type Application struct {
	neuralNetwork *gocv.Net
	layersNames   []string
	decoder       *decoder.H264Decoder
	settings      *AppSettings
}

func NewApp(settings *AppSettings) (*Application, error) {
	neuralNet := gocv.ReadNet(settings.NeuralNetworkSettings.DarknetWeights, settings.NeuralNetworkSettings.DarknetCFG)
	yoloLayersIdx := neuralNet.GetUnconnectedOutLayers()
	outLayerNames := make([]string, 0, 3)

	for _, idx := range yoloLayersIdx {
		layer := neuralNet.GetLayer(idx)
		outLayerNames = append(outLayerNames, layer.GetName())
	}

	if err := neuralNet.SetPreferableBackend(gocv.ParseNetBackend(settings.NeuralNetworkSettings.Backend)); err != nil {
		return nil, errors.Wrapf(err, "Can't set backend %s", settings.NeuralNetworkSettings.Backend)
	}

	if err := neuralNet.SetPreferableTarget(gocv.ParseNetTarget(settings.NeuralNetworkSettings.Target)); err != nil {
		return nil, errors.Wrapf(err, "Can't set target %s", settings.NeuralNetworkSettings.Target)
	}

	d, err := decoder.New(decoder.PixelFormatBGR)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create H264 decoder")
	}

	return &Application{
		neuralNetwork: &neuralNet,
		layersNames:   outLayerNames,
		decoder:       d,
		settings:      settings,
	}, nil
}

// StartMJPEGStream Start MJPEG video stream in separate goroutine
func (app *Application) StartMJPEGStream() *mjpeg.Stream {
	stream := mjpeg.NewStream()

	go func() {
		fmt.Printf("Starting MJPEG on http://localhost:%d\n", app.settings.MjpegSettings.Port)

		router := mux.NewRouter()
		c := cors.New(cors.Options{
			AllowedOrigins:   []string{"*"},
			AllowCredentials: true,
		})

		router.HandleFunc("/", stream.ServeHTTP)
		http.Handle("/", c.Handler(router))

		if err := http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", app.settings.MjpegSettings.Port), nil); err != nil {
			log.Fatalln(err)
		}
	}()
	return stream
}

func (app *Application) Run() error {
	settings := app.settings

	/* Open imshow() GUI in needed */
	var window *gocv.Window
	if settings.MjpegSettings.ImshowEnable {
		fmt.Println("Press 'ESC' to stop imshow()")
		window = gocv.NewWindow("ML")
		window.ResizeWindow(settings.VideoSettings.ReducedWidth, settings.VideoSettings.ReducedHeight)
		defer window.Close()
	}

	/* Initialize MJPEG server if needed */
	var stream *mjpeg.Stream
	if settings.MjpegSettings.Enable {
		stream = app.StartMJPEGStream()
	}

	var videoCapture *gocv.VideoCapture
	var err error
	var pc net.PacketConn
	if app.settings.Source == "webcam" {
		fmt.Println("Starting to capture webcam")
		videoCapture, err = gocv.VideoCaptureDevice(app.settings.VideoCaptureDeviceSettings.DeviceID)
		if err != nil {
			return errors.Wrap(err, "Can't open video capture")
		}
	} else if app.settings.Source == "video" {
		fmt.Println("Starting to capture video")
		videoCapture, err = gocv.OpenVideoCapture(settings.VideoSettings.Source)
		if err != nil {
			return errors.Wrap(err, "Can't open video capture")
		}
	} else if app.settings.Source == "camera" {
		fmt.Println("Starting to listen for packets")
		pc, err = reuseable.ListenPacket("udp4", fmt.Sprintf("%s:%d", app.settings.CameraSettings.Address, app.settings.CameraSettings.Port))
		if err != nil {
			return errors.Wrap(err, "Can't open video capture")
		}
	}

	fmt.Println("Ready to receive data")

	/* Prepare frame */
	img := NewFrameData()
	buf := make([]byte, 1514)

	/* Read frames in a */
	for {
		// Grab a frame
		if videoCapture != nil {
			if ok := videoCapture.Read(&img.ImgSource); !ok {
				fmt.Println("Can't read next frame, stop grabbing...")
				break
			}
		} else if pc != nil {
			fmt.Println("Read from pc")
			n, _, err := pc.ReadFrom(buf)
			if err != nil {
				return fmt.Errorf("failed to read from buffer: %w", err)
			}

			if n < 72 {
				fmt.Println("Empty frame has been loaded. Sleep for 400 ms")
				time.Sleep(400 * time.Millisecond)
				continue
			}

			frames, err := app.decoder.Decode(buf[72:n])
			if err != nil {
				fmt.Println("Failed to decode frame. Sleep for 400 ms")
				time.Sleep(400 * time.Millisecond)
				continue
			}

			if len(frames) == 0 {
				continue
			}

			fmt.Println(frames[0].Width)
			fmt.Println(frames[0].Height)

			if err := img.Load(frames[0]); err != nil {
				return fmt.Errorf("failed to load image")
			}
		}

		/* Skip empty frame */
		if img.ImgSource.Empty() {
			fmt.Println("Empty frame has been detected. Sleep for 400 ms")
			time.Sleep(400 * time.Millisecond)
			continue
		}

		/* Scale frame */
		var width, height int
		if settings.Source == "camera" && settings.CameraSettings.ReducedWidth != settings.CameraSettings.Width && settings.CameraSettings.ReducedHeight != settings.CameraSettings.Height {
			width = settings.CameraSettings.ReducedWidth
			height = settings.CameraSettings.ReducedHeight
		} else {
			width = settings.VideoSettings.ReducedWidth
			height = settings.VideoSettings.ReducedHeight
		}
		if err := img.Preprocess(width, height); err != nil {
			fmt.Printf("Can't preprocess. Error: %s. Sleep for 400ms\n", err.Error())
			time.Sleep(400 * time.Millisecond)
			continue
		}

		/* Detection */
		if settings.NeuralNetworkSettings.Enable {
			detected := app.performDetectionSequential(img, settings.NeuralNetworkSettings.NetClasses, settings.NeuralNetworkSettings.TargetClasses)
			if len(detected) != 0 {
				for _, detection := range detected {
					c := colors[detection.ClassID%len(colors)]

					FixRectForOpenCV(&detection.Rect, settings.CameraSettings.Width, settings.CameraSettings.Height)
					gocv.Rectangle(&img.ImgScaled, detection.Rect, c, 1)
					gocv.PutText(&img.ImgScaled, detection.ClassName, image.Pt(detection.Rect.Min.X, detection.Rect.Min.Y), gocv.FontHersheyPlain, 1.0, c, 1)
				}
			}
		}

		if settings.MjpegSettings.ImshowEnable {
			window.IMShow(img.ImgScaled)
			if window.WaitKey(1) == 27 {
				break
			}
		}

		if settings.MjpegSettings.Enable {
			buf, err := gocv.IMEncode(".jpg", img.ImgScaled)
			if err != nil {
				log.Printf("Error while decoding to JPG (mjpeg): %s", err.Error())
			} else {
				stream.UpdateJPEG(buf.GetBytes())
			}
		}
	}

	// Hard release memory
	img.Close()
	app.Close()
	pc.Close()

	return nil
}

func (app *Application) performDetectionSequential(frame *FrameData, netClasses, targetClasses []string) []*DetectedObject {
	detectedRects, err := DetectObjects(app, frame.ImgScaledCopy, netClasses, targetClasses...)
	if err != nil {
		log.Printf("Can't detect objects on provided image due the error: %s. Sleep for 100ms", err.Error())
		frame.ImgScaledCopy.Close()
		time.Sleep(100 * time.Millisecond)
	}
	frame.ImgScaledCopy.Close() // free the memory
	return detectedRects
}

// Close Free memory for underlying objects
func (app *Application) Close() {
	app.neuralNetwork.Close()

	if app.decoder != nil {
		app.decoder.Close()
	}
}
