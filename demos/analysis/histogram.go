package main

import (
	"fmt"
	"math"

	"github.com/unixpickle/sgd"
)

func createHistogram(net sgd.Learner) {
	const (
		step     = 0.002
		numSteps = 100
	)

	fmt.Println("Creating weight magnitude histogram:")
	count := map[int]int{}
	for _, p := range net.Parameters() {
		for _, x := range p.Vector {
			bin := int(math.Floor(x/step + 0.5))
			count[bin]++
		}
	}
	fmt.Print("x = [")
	for i := -numSteps; i <= numSteps; i++ {
		fmt.Printf(" %f", float64(i)*step)
	}
	fmt.Println(" ];")
	fmt.Print("y = [")
	for i := -numSteps; i <= numSteps; i++ {
		fmt.Printf(" %d", count[i])
	}
	fmt.Println(" ];")
}
