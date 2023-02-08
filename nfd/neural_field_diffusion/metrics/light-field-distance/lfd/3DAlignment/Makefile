CC=gcc
CFLAGS=-g
INCLUDES=-I. -I./fftw
MAIN=3DAlignment
LDLIBS=-lglut -lGL -lGLU -lm -lOSMesa

SRCS=$(wildcard *.c) $(wildcard ./fftw/*.c)

OBJS=$(SRCS:.c=.o)

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))


.PRECIOUS: %(OBJS)

all:	$(MAIN)
$(MAIN):	$(OBJS)
	$(CC) $(CLFAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LDLIBS)


release: $(MAIN)
	mv $(MKFILE_DIR)/$(MAIN) $(MKFILE_DIR)/../Executable

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean


clean:
	$(RM) $(OBJS) 