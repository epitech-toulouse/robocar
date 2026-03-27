#pragma once

/**
 * @file heading_alignment.hpp
 * @brief GPS bearing/distance/angular error calculations.
 *
 * All angles in degrees.
 * Uses Haversine formula for distance and bearing computations.
 */

#include "nav_types.h"

/**
 * @brief Compute bearing from current position to target.
 * @return Bearing in degrees 0..360 (0=North, 90=East).
 */
float compute_bearing(float lat1, float lon1, float lat2, float lon2);

/**
 * @brief Compute distance between two coordinates using Haversine.
 * @return Distance in meters.
 */
float compute_distance(float lat1, float lon1, float lat2, float lon2);

/**
 * @brief Compute signed angular error: how much to turn to face the target.
 * @param heading  Current heading 0..360°.
 * @param bearing  Bearing to target 0..360°.
 * @return Error in degrees -180..+180. Positive = turn right, negative = turn left.
 */
float compute_angular_error(float heading, float bearing);

/**
 * @brief Full alignment computation.
 */
AlignmentResult compute_alignment(float cur_lat, float cur_lon,
                                  float cur_heading,
                                  float tgt_lat, float tgt_lon);
