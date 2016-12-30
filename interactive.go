// +build !js

package leea

import (
	"fmt"
	"os"
	"os/signal"
	"sync/atomic"
)

// loopUntilKilled continually calls tf until sf returns
// false or until the user sends a kill signal.
//
// This is borrowed from
// https://github.com/unixpickle/sgd/blob/0e3d4c9d317b1095d02febdaedf802f6d1dbd5b1/interactive.go.
func loopUntilKilled(sf func() bool, tf func()) {
	var killed uint32

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	defer func() {
		select {
		case <-c:
		default:
			signal.Stop(c)
			close(c)
		}
	}()

	go func() {
		_, ok := <-c
		if !ok {
			return
		}
		signal.Stop(c)
		close(c)
		atomic.StoreUint32(&killed, 1)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
	}()

	for atomic.LoadUint32(&killed) == 0 {
		if sf != nil {
			if !sf() {
				return
			}
		}
		if atomic.LoadUint32(&killed) != 0 {
			return
		}
		tf()
	}
}
