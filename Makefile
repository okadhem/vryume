PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# =========================
# Compiler & Linker
# =========================
CXX := $(NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
LD  := clang++
LIBS_PATH := -L$(PROJECT_ROOT)third_party/libs
LDFLAGS := -shared $(LIBS_PATH) -landroid -llog -lvulkan -lopenxr_loader

# =========================
# Project structure
# =========================
SRC_DIR := src
OBJ_DIR := build
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))
TARGET := libengine.so

# =========================
# Common flags
# =========================
INCLUDES_PATH = -I$(PROJECT_ROOT)third_party/includes
COMMON_FLAGS := -std=c++20 \
                -fno-exceptions \
                -fno-rtti \
                -nostdlib++ \
                -fPIC \
				-x c++ \
				$(INCLUDES_PATH)

# =========================
# Configurations
# =========================
DEBUG_FLAGS := -O0 -g -Wall
RELEASE_FLAGS := -O3 -DNDEBUG

# =========================
# Default target
# =========================
all: debug

# =========================
# Build rules
# =========================
# Debug
debug: CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
debug: $(OBJ_DIR) $(OBJ)
	$(CXX) $(OBJ) -o ./build/libengine_debug.so $(LDFLAGS)

# Release
release: CXXFLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
release: $(OBJ_DIR) $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

# =========================
# Object build rule
# =========================
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# =========================
# Build directory
# =========================
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# =========================
# Clean
# =========================
clean:
	rm -rf $(OBJ_DIR) $(TARGET) libengine_debug.so

