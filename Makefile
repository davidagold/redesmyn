SHELL = /bin/bash

.PHONY: run

run:
	RUST_LOG=INFO cargo run
