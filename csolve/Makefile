CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Iinclude
LDFLAGS ?=
LDLIBS  ?= -lm

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)
DEP := $(OBJ:.o=.d)
BIN := csolve

.PHONY: all clean test

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS)

# Автоматические зависимости от заголовков: изменение include/csolve.h
# пересобирает все объектные файлы.
src/%.o: src/%.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

tests/unit_tests: tests/unit_tests.c $(filter-out src/main.o,$(OBJ))
	$(CC) $(CFLAGS) -o $@ tests/unit_tests.c $(filter-out src/main.o,$(OBJ)) \
		$(LDFLAGS) $(LDLIBS)

test: tests/unit_tests
	./tests/unit_tests

clean:
	rm -f $(OBJ) $(DEP) $(BIN) tests/unit_tests
	rm -rf tests/unit_tests.dSYM

-include $(DEP)
