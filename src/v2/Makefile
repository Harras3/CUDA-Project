CC = nvcc
CFLAGS = -O2
LDFLAGS = -lm

EXE = nn_cuda
SRC = nn.cu

all: $(EXE) run profile

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) $(LDFLAGS)

run: $(EXE)
	@echo "Running CUDA application and measuring execution time..."
        ./$(EXE)

profile: $(EXE)
	@echo "Profiling with nvprof..."
	nvprof --log-file nvprof.txt ./$(EXE)
	@echo "Profiling with nsys..."
	nsys profile --stats=true --output=nsys_report ./$(EXE)
	@echo "Profiles saved to nvprof.txt and nsys_report.qdrep"

clean:
	rm -f $(EXE) nvprof.txt nsys_report.qdrep nsys_report.sqlite 
