#pragma once

/**
 * @file gps_reader.hpp
 * @brief USB Host CDC GPS reader for ESP32-S3.
 *
 * Reads NMEA sentences from a GPS module connected via USB (CDC-ACM).
 * Parses $GNGGA/$GPGGA (position, fix, sats) and $GNRMC/$GPRMC (heading, speed).
 * Provides a thread-safe GpsFix struct to the navigation controller.
 */

#include "nav_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize USB Host stack and start GPS reader task.
 * Must be called from app_main() before the navigation loop.
 */
void init_gps_usb(void);

/**
 * @brief Get the latest GPS fix (thread-safe).
 * @param[out] fix  Filled with the latest GPS data.
 * @return true if a valid fix is available and not stale.
 */
bool get_gps_fix(GpsFix &fix);

#ifdef __cplusplus
}
#endif
