PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Project structure
SRC_DIR := src
OBJ_DIR := build
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))

SHADER_OBJ_DIR := $(OBJ_DIR)/assets
COMP_SHADERS_SRC := $(wildcard $(SRC_DIR)/*.comp)
COMP_SHADERS_OBJ := $(patsubst $(SRC_DIR)/%.comp,$(SHADER_OBJ_DIR)/%.spv,$(COMP_SHADERS_SRC))

# Shader compilation
GLSLCFLAGS :=
GLSLC := glslc


# Common flags
INCLUDES_PATH = -I$(PROJECT_ROOT)third_party/includes
COMMON_FLAGS := -std=c++20 \
                -fno-exceptions \
                -fno-rtti \
                -nostdlib++ \
                -fPIC \
				-x c++ \
				-fno-omit-frame-pointer \
				$(INCLUDES_PATH)

DEBUG_FLAGS := -O0 -g -Wall
RELEASE_FLAGS := -O3 -DNDEBUG


# Default target
all: desktop_debug


# Build rules

android_debug: CXX := $(NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
android_debug: LD  := clang++
android_debug: LIBS_PATH := -L$(PROJECT_ROOT)third_party/android/libs
android_debug: LDFLAGS := -shared $(LIBS_PATH) -landroid -llog -lvulkan -lopenxr_loader
android_debug: CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS) -DXR_USE_PLATFORM_ANDROID
android_debug: $(OBJ_DIR) $(SHADER_OBJ_DIR) $(OBJ) $(COMP_SHADERS_OBJ) 
	$(CXX) $(OBJ) -o ./build/libengine_debug.so $(LDFLAGS)

desktop_debug: CXX := clang++
desktop_debug: LD  := clang++
desktop_debug: LIBS_PATH := -L$(PROJECT_ROOT)third_party/linux/libs
desktop_debug: LDFLAGS := $(LIBS_PATH) -lvulkan -lopenxr_loader -ldl
desktop_debug: CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
desktop_debug: $(OBJ_DIR) $(SHADER_OBJ_DIR) $(OBJ) $(COMP_SHADERS_OBJ) 
	$(CXX) $(OBJ) -o ./build/game_exe $(LDFLAGS)



# Object build rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SHADER_OBJ_DIR)/%.spv: $(SRC_DIR)/%.comp $(SRC_DIR)/*.h.glsl
	$(GLSLC) $(GLSLCFLAGS) -c $< -o $@


# Build directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(SHADER_OBJ_DIR):
	mkdir -p $(SHADER_OBJ_DIR)

# Clean
clean:
	rm -rf $(OBJ_DIR)

