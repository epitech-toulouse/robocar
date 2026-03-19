#ifndef VESC_LIDAR_UART_DRIVER_H
#define VESC_LIDAR_UART_DRIVER_H

#if __cplusplus
extern "C" {
#endif

void init_lidar_uart(void);
void delete_lidar_uart(void);

void init_vesc_rmt_uart(void);
void delete_vesc_rmt_uart(void);

#if __cplusplus
}
#endif

#endif /* VESC_LIDAR_UART_DRIVER_H */
