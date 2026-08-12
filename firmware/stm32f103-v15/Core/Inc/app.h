#ifndef APP_H
#define APP_H

#include "stm32f1xx_hal.h"

void App_Init(UART_HandleTypeDef *command_uart);
void App_Task(void);

/* Call these from the matching STM32 HAL weak callbacks in main.c. */
void App_UartRxCpltCallback(UART_HandleTypeDef *huart);
void App_UartErrorCallback(UART_HandleTypeDef *huart);

#endif
