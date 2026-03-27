#pragma once

/**
 * @file obstacle_avoidance.hpp
 * @brief LIDAR-based obstacle analysis for navigation.
 *
 * Analyzes LIDAR scan sectors and produces an ObstacleResult
 * without commanding motors directly.
 */

#include "nav_types.h"
#include "lidarReader.hpp"
#include <vector>

/**
 * @brief Analyze LIDAR scan for obstacles in front, left, and right sectors.
 * @param scan  Latest 360° LIDAR scan points.
 * @return ObstacleResult with sector distances and suggested avoidance.
 */
ObstacleResult analyze_obstacles(const std::vector<LidarPoint> &scan);
