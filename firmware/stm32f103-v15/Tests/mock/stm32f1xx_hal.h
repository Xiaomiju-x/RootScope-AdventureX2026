#ifndef STM32F1XX_HAL_H
#define STM32F1XX_HAL_H

#include <stdint.h>

typedef struct
{
    uint32_t id;
    uint32_t ODR;
} GPIO_TypeDef;
typedef struct { uint32_t id; } UART_HandleTypeDef;

typedef enum
{
    GPIO_PIN_RESET = 0U,
    GPIO_PIN_SET
} GPIO_PinState;

typedef enum
{
    HAL_OK = 0x00U,
    HAL_ERROR = 0x01U,
    HAL_BUSY = 0x02U
} HAL_StatusTypeDef;

extern GPIO_TypeDef mock_gpio_a;
extern GPIO_TypeDef mock_gpio_b;

#define GPIOA (&mock_gpio_a)
#define GPIOB (&mock_gpio_b)

#define GPIO_PIN_0  ((uint16_t)0x0001U)
#define GPIO_PIN_1  ((uint16_t)0x0002U)
#define GPIO_PIN_2  ((uint16_t)0x0004U)
#define GPIO_PIN_3  ((uint16_t)0x0008U)
#define GPIO_PIN_4  ((uint16_t)0x0010U)
#define GPIO_PIN_5  ((uint16_t)0x0020U)
#define GPIO_PIN_6  ((uint16_t)0x0040U)
#define GPIO_PIN_7  ((uint16_t)0x0080U)
#define GPIO_PIN_8  ((uint16_t)0x0100U)

void HAL_GPIO_WritePin(GPIO_TypeDef *port, uint16_t pin,
                       GPIO_PinState state);
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *huart, uint8_t *data,
                                    uint16_t length, uint32_t timeout);
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *huart, uint8_t *data,
                                      uint16_t length);
uint32_t HAL_GetTick(void);

#endif
