# Makefile for FlashAttention Metal Project

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -fobjc-arc
LDFLAGS = -framework Metal -framework Foundation -framework CoreGraphics

TARGET = flash_attn
SRC = main.mm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
