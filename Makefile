.PHONY: default
default: clean compile_commands.json hello_world_cuda

.PHONY: clean
clean:
	rm -f hello_world_cuda compile_commands.json

compile_commands.json:
	bear -- $(MAKE) hello_world_cuda

hello_world_cuda:
	nvcc main.cu -o hello_world_cuda
