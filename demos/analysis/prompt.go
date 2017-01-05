package main

import (
	"fmt"
	"os"
)

func promptChoice(msg string) bool {
	for {
		fmt.Print(msg + " [y/n]: ")
		var line string
		for {
			var next [1]byte
			if n, err := os.Stdin.Read(next[:]); err != nil {
				panic(err)
			} else if n != 0 {
				if next[0] == '\n' {
					break
				} else if next[0] != '\r' {
					line += string(next[:])
				}
			}
		}
		if line == "y" {
			return true
		} else if line == "n" {
			return false
		} else {
			fmt.Println("Unrecognized input:", line)
		}
	}
}
