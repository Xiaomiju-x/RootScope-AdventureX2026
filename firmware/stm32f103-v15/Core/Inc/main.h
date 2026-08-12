/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define Z_INA_Pin GPIO_PIN_0
#define Z_INA_GPIO_Port GPIOA
#define Z_INB_Pin GPIO_PIN_1
#define Z_INB_GPIO_Port GPIOA
#define Z_INC_Pin GPIO_PIN_2
#define Z_INC_GPIO_Port GPIOA
#define Z_IND_Pin GPIO_PIN_3
#define Z_IND_GPIO_Port GPIOA
#define UNUSED_PA4_Pin GPIO_PIN_4
#define UNUSED_PA4_GPIO_Port GPIOA
#define UNUSED_PA5_Pin GPIO_PIN_5
#define UNUSED_PA5_GPIO_Port GPIOA
#define UNUSED_PA6_Pin GPIO_PIN_6
#define UNUSED_PA6_GPIO_Port GPIOA
#define UNUSED_PA7_Pin GPIO_PIN_7
#define UNUSED_PA7_GPIO_Port GPIOA
#define PUMP_Pin GPIO_PIN_6
#define PUMP_GPIO_Port GPIOB
/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
