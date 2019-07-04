.PHONY: help
help: Makefile
	@ sed -n 's/^##//p' $<

## clean_logs: Delete all logs
clean_logs :
	rm ./logs/*

## clean_weights: Delete all the weights
clean_weights :
	rm ./weights/*
