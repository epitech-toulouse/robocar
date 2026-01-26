/**
 * LidarParser - C++ LD19 LiDAR Parser for Robocar
 * 
 * Thread-safe parser that continuously reads from serial port
 * and maintains a buffer of parsed points.
 */

#ifndef LIDAR_PARSER_HPP
#define LIDAR_PARSER_HPP

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Point structure for C API
typedef struct {
    float angle;      // degrees (0-360)
    float distance;   // meters
} LidarPoint;

// Opaque handle type
typedef void* LidarParserHandle;

// C API for Python ctypes bindings
LidarParserHandle lidar_parser_create(const char* port, int baudrate, float angle_offset);
void lidar_parser_destroy(LidarParserHandle handle);
int lidar_parser_get_points(LidarParserHandle handle, LidarPoint* buffer, int max_points);
void lidar_parser_stop(LidarParserHandle handle);
int lidar_parser_is_connected(LidarParserHandle handle);

#ifdef __cplusplus
}
#endif

#endif // LIDAR_PARSER_HPP
