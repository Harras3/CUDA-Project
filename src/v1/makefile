CC = gcc
CFLAGS = -Wall -O2 -pg

EXE = nn.exe
SRC = nn.c

all: $(EXE) run profile

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	@echo "Running application and measuring execution time..."
	@time ./$(EXE)

profile: $(EXE)
	@echo "Profiling with gprof..."
	./$(EXE)
	gprof $(EXE) gmon.out > profile.txt
	@echo "Profile saved to profile.txt"

clean:
	rm -f $(EXE) gmon.out profile.txt
