package main

import (
	"flag"
	"fmt"
	"log"
	_ "net/http/pprof"
	"time"

	"gocv.io/x/gocv"

	"github.com/genert/ml"
)

func main() {
	settingsFile := flag.String("settings", "config.json", "Path to application's settings")
	flag.Parse()

	fmt.Printf("gocv version: %s\n", gocv.Version())
	fmt.Printf("opencv lib version: %s\n", gocv.OpenCVVersion())

	/* Read settings */
	settings, err := ml.NewSettings(*settingsFile)
	if err != nil {
		log.Println(err)
		return
	}

	app, err := ml.NewApp(settings)
	if err != nil {
		log.Println(err)
		return
	}
	defer app.Close()

	if err := app.Run(); err != nil {
		log.Println(err)
	}

	fmt.Println("Shutting down...")
	time.Sleep(2 * time.Second) // @todo temporary fix: need to wait a bit time for last call of neuralNet.Detect(...)

	// Hard release memory
	app.Close()
}
