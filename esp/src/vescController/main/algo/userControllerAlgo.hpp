#ifndef USER_CONTROLLER_ALGO_HPP
#define USER_CONTROLLER_ALGO_HPP

#include <cmath>

#include "api/driving_algorithm_interface.hpp"
#include "api/user_controller_api.hpp"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

class UserControllerAlgo : public DrivingAlgorithmApi
{
public:
    explicit UserControllerAlgo(UserControllerApi &userController)
        : _userController(userController)
    {
    }
    ~UserControllerAlgo() = default;

    bool compute(DrivingAlgorithmOutput &output) override;
    float getPriority() override;
    bool available(void) override { return _userController.isConnected(); }

private:
    UserControllerApi &_userController;


    static constexpr TickType_t logPeriodTicks = pdMS_TO_TICKS(500);
    const char *const tag = "UserControllerAlgo";
    TickType_t lastLogTick = 0;
};

#endif /* USER_CONTROLLER_ALGO_HPP */
