CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17

TARGET_SERVER = server
TARGET_CLIENT = client
TARGET_VIS = visualizer

SRC_SERVER = server.cpp
SRC_CLIENT = client.cpp
SRC_VIS = visualizer.cpp

SFML_FLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lpthread

all: $(TARGET_SERVER) $(TARGET_CLIENT) $(TARGET_VIS)

$(TARGET_SERVER): $(SRC_SERVER)
	$(CXX) $(CXXFLAGS) -o $(TARGET_SERVER) $(SRC_SERVER)

$(TARGET_CLIENT): $(SRC_CLIENT)
	$(CXX) $(CXXFLAGS) -o $(TARGET_CLIENT) $(SRC_CLIENT)

$(TARGET_VIS): $(SRC_VIS)
	$(CXX) $(CXXFLAGS) -o $(TARGET_VIS) $(SRC_VIS) $(SFML_FLAGS)

clean:
	rm -f $(TARGET_SERVER) $(TARGET_CLIENT) $(TARGET_VIS)

fclean: clean

re: fclean all
