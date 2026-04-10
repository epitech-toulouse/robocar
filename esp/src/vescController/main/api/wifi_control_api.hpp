/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** wifi control interface
*/

#ifndef WIFI_CONTROL_API_HPP
#define WIFI_CONTROL_API_HPP

class WifiControlApi {
public:
    virtual ~WifiControlApi() = default;

    virtual void start(void) = 0;
    virtual void stop(void) = 0;
    virtual bool isActivated(void) = 0;
    virtual bool getManualControl(float &duty, float &steer, bool &emergency) = 0;
};

WifiControlApi &wifiControlServer();

#endif /* WIFI_CONTROL_API_HPP */