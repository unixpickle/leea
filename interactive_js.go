// +build js

package leea

func loopUntilKilled(sf func() bool, tf func()) {
	for {
		if !sf() {
			return
		}
		tf()
	}
}
