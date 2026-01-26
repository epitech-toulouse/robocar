"""C++ LiDAR Parser wrapper using ctypes

Drop-in replacement for LidarParser with C++ backend for better performance.
"""

import ctypes
import os
from pathlib import Path

# Point structure matching C struct
class LidarPointC(ctypes.Structure):
    _fields_ = [
        ("angle", ctypes.c_float),
        ("distance", ctypes.c_float),
    ]

class LidarParser:
    def __init__(self, port=['/dev/ttyUSB0', '/dev/ttyUSB1'], baudrate=230400, angle_offset=15.0):
        # Find and load the shared library - check multiple locations
        search_paths = [
            Path(__file__).parent / "cpp" / "liblidar_parser.so",
            Path(__file__).parent / "cpp" / "build" / "liblidar_parser.so",
        ]
        
        lib_path = None
        for path in search_paths:
            if path.exists():
                lib_path = path
                break
        
        if lib_path is None:
            raise RuntimeError(f"Could not find liblidar_parser.so in: {[str(p) for p in search_paths]}")
        
        self._lib = ctypes.CDLL(str(lib_path))
        
        # Setup function signatures
        self._lib.lidar_parser_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
        self._lib.lidar_parser_create.restype = ctypes.c_void_p
        
        self._lib.lidar_parser_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lidar_parser_destroy.restype = None
        
        self._lib.lidar_parser_get_points.argtypes = [ctypes.c_void_p, ctypes.POINTER(LidarPointC), ctypes.c_int]
        self._lib.lidar_parser_get_points.restype = ctypes.c_int
        
        self._lib.lidar_parser_stop.argtypes = [ctypes.c_void_p]
        self._lib.lidar_parser_stop.restype = None
        
        self._lib.lidar_parser_is_connected.argtypes = [ctypes.c_void_p]
        self._lib.lidar_parser_is_connected.restype = ctypes.c_int
        
        # Use first port from list
        port_str = port[0] if isinstance(port, list) else port
        
        # Create parser instance
        self._handle = self._lib.lidar_parser_create(
            port_str.encode('utf-8'),
            baudrate,
            angle_offset
        )
        
        # Buffer for receiving points
        self._max_points = 2000
        self._point_buffer = (LidarPointC * self._max_points)()
    
    def get_points(self):
        """Returns list of points as [{'angle': float, 'distance': float}, ...]"""
        if not self._handle:
            return []
        
        count = self._lib.lidar_parser_get_points(
            self._handle,
            self._point_buffer,
            self._max_points
        )
        
        return [
            {'angle': self._point_buffer[i].angle, 'distance': self._point_buffer[i].distance}
            for i in range(count)
        ]
    
    def is_connected(self):
        """Returns True if connected to serial port"""
        if not self._handle:
            return False
        return self._lib.lidar_parser_is_connected(self._handle) == 1
    
    def stop(self):
        """Stop the parser and release resources"""
        if self._handle:
            self._lib.lidar_parser_stop(self._handle)
            self._lib.lidar_parser_destroy(self._handle)
            self._handle = None
    
    def __del__(self):
        if hasattr(self, '_handle'):
            self.stop()
