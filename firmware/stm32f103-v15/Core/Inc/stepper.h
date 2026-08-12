#ifndef STEPPER_H
#define STEPPER_H

#include "stm32f1xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    STEPPER_STAGE_IDLE = 0,
    STEPPER_STAGE_PRECHARGE,
    STEPPER_STAGE_RUN,
    STEPPER_STAGE_HOLD
} StepperStage_t;

typedef struct
{
    GPIO_TypeDef *port[4];
    uint16_t pin[4];
    int8_t wiring_direction;
    int8_t move_direction;
    uint8_t phase;
    uint8_t sequence_mode;
    uint32_t steps_remaining;
    uint32_t target_interval_ms;
    uint32_t current_interval_ms;
    uint32_t next_step_ms;
    uint32_t ramp_step_counter;
    StepperStage_t stage;
    bool busy;
} Stepper_t;

void Stepper_Init(Stepper_t *motor,
                  GPIO_TypeDef *ina_port, uint16_t ina_pin,
                  GPIO_TypeDef *inb_port, uint16_t inb_pin,
                  GPIO_TypeDef *inc_port, uint16_t inc_pin,
                  GPIO_TypeDef *ind_port, uint16_t ind_pin,
                  int8_t wiring_direction);

bool Stepper_Move(Stepper_t *motor, int32_t signed_steps,
                  uint32_t interval_ms, uint32_t now_ms);
void Stepper_Task(Stepper_t *motor, uint32_t now_ms);
void Stepper_Stop(Stepper_t *motor);
void Stepper_SetRawPattern(Stepper_t *motor, uint8_t pattern);
bool Stepper_SetSequenceMode(Stepper_t *motor, uint8_t sequence_mode);
bool Stepper_IsBusy(const Stepper_t *motor);

#endif
