CC=gcc
CFLAGS=-I. -g -Wall
DEPS=ds.h
TARGET=Distance

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

all: $(TARGET)

$(TARGET): Main.c ds.h
	$(CC) $(CFLAGS) -o $(TARGET) Main.c 

release: $(TARGET)
	mv $(MKFILE_DIR)/$(TARGET) $(MKFILE_DIR)/../Executable
