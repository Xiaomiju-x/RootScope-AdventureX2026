#include "stepper.h"
#include "app_config.h"

/*
 * Diagnostic sequences for the three possible four-phase cycles.  Bit 0..3
 * always maps directly to INA..IND, so no physical rewiring is required.
 *
 * Modes 0..2 energize one phase at a time to minimize supply current.
 * Modes 3..5 energize two adjacent magnetic phases for higher torque.
 *
 *   0: A-B-C-D wave       3: A-B-C-D dual
 *   1: A-C-B-D wave       4: A-C-B-D dual
 *   2: A-B-D-C wave       5: A-B-D-C dual
 */
static const uint8_t step_sequences[6][4] =
{
    {0x01U, 0x02U, 0x04U, 0x08U},
    {0x01U, 0x04U, 0x02U, 0x08U},
    {0x01U, 0x02U, 0x08U, 0x04U},
    {0x03U, 0x06U, 0x0CU, 0x09U},
    {0x05U, 0x06U, 0x0AU, 0x09U},
    {0x03U, 0x0AU, 0x0CU, 0x05U}
};

static void Stepper_WritePattern(Stepper_t *motor, uint8_t pattern)
{
    uint8_t i;

    pattern &= 0x0FU;
    for (i = 0U; i < 4U; ++i)
    {
        HAL_GPIO_WritePin(motor->port[i], motor->pin[i],
                          ((pattern & (1U << i)) != 0U) ?
                          GPIO_PIN_SET : GPIO_PIN_RESET);
    }
}

static void Stepper_WritePhase(Stepper_t *motor, uint8_t phase)
{
    Stepper_WritePattern(
        motor, step_sequences[motor->sequence_mode][phase & 0x03U]);
}

static void Stepper_Release(Stepper_t *motor)
{
    Stepper_WritePattern(motor, 0U);
}

void Stepper_Init(Stepper_t *motor,
                  GPIO_TypeDef *ina_port, uint16_t ina_pin,
                  GPIO_TypeDef *inb_port, uint16_t inb_pin,
                  GPIO_TypeDef *inc_port, uint16_t inc_pin,
                  GPIO_TypeDef *ind_port, uint16_t ind_pin,
                  int8_t wiring_direction)
{
    motor->port[0] = ina_port;
    motor->port[1] = inb_port;
    motor->port[2] = inc_port;
    motor->port[3] = ind_port;
    motor->pin[0] = ina_pin;
    motor->pin[1] = inb_pin;
    motor->pin[2] = inc_pin;
    motor->pin[3] = ind_pin;
    motor->wiring_direction = (wiring_direction >= 0) ? 1 : -1;
    motor->move_direction = 1;
    motor->phase = 0U;
    motor->sequence_mode = 3U;
    motor->steps_remaining = 0U;
    motor->target_interval_ms = APP_MIN_STEP_INTERVAL_MS;
    motor->current_interval_ms = STEPPER_START_INTERVAL_MS;
    motor->next_step_ms = 0U;
    motor->ramp_step_counter = 0U;
    motor->stage = STEPPER_STAGE_IDLE;
    motor->busy = false;
    Stepper_Release(motor);
}

bool Stepper_Move(Stepper_t *motor, int32_t signed_steps,
                  uint32_t interval_ms, uint32_t now_ms)
{
    uint32_t magnitude;

    if (motor->busy || (signed_steps == 0) || (signed_steps == INT32_MIN))
    {
        return false;
    }

    magnitude = (signed_steps < 0) ?
                (uint32_t)(-signed_steps) : (uint32_t)signed_steps;
    motor->move_direction = ((signed_steps > 0) ? 1 : -1) *
                            motor->wiring_direction;
    motor->steps_remaining = magnitude;
    motor->target_interval_ms =
        (interval_ms < APP_MIN_STEP_INTERVAL_MS) ?
        APP_MIN_STEP_INTERVAL_MS : interval_ms;
    motor->current_interval_ms =
        (motor->target_interval_ms < STEPPER_START_INTERVAL_MS) ?
        STEPPER_START_INTERVAL_MS : motor->target_interval_ms;
    motor->ramp_step_counter = 0U;
    motor->stage = STEPPER_STAGE_PRECHARGE;
    Stepper_WritePhase(motor, motor->phase);
    motor->next_step_ms = now_ms + STEPPER_PRECHARGE_MS;
    motor->busy = true;
    return true;
}

void Stepper_Task(Stepper_t *motor, uint32_t now_ms)
{
    if (!motor->busy || ((int32_t)(now_ms - motor->next_step_ms) < 0))
    {
        return;
    }

    if (motor->stage == STEPPER_STAGE_HOLD)
    {
        motor->steps_remaining = 0U;
        motor->stage = STEPPER_STAGE_IDLE;
        motor->busy = false;
        Stepper_Release(motor);
        return;
    }

    motor->stage = STEPPER_STAGE_RUN;
    if (motor->move_direction > 0)
    {
        motor->phase = (uint8_t)((motor->phase + 1U) & 0x03U);
    }
    else
    {
        motor->phase = (uint8_t)((motor->phase + 3U) & 0x03U);
    }
    Stepper_WritePhase(motor, motor->phase);
    --motor->steps_remaining;

    if (motor->steps_remaining == 0U)
    {
        /*
         * Keep the last dual-phase vector energized long enough for the
         * loaded rotor to reach the commanded tooth before releasing it.
         */
        motor->stage = STEPPER_STAGE_HOLD;
        motor->next_step_ms = now_ms + STEPPER_FINAL_HOLD_MS;
        return;
    }

    if (motor->current_interval_ms > motor->target_interval_ms)
    {
        ++motor->ramp_step_counter;
        if (motor->ramp_step_counter >= STEPPER_RAMP_EVERY_STEPS)
        {
            --motor->current_interval_ms;
            motor->ramp_step_counter = 0U;
        }
    }
    motor->next_step_ms = now_ms + motor->current_interval_ms;
}

void Stepper_Stop(Stepper_t *motor)
{
    motor->steps_remaining = 0U;
    motor->stage = STEPPER_STAGE_IDLE;
    motor->busy = false;
    Stepper_Release(motor);
}

void Stepper_SetRawPattern(Stepper_t *motor, uint8_t pattern)
{
    motor->steps_remaining = 0U;
    motor->stage = STEPPER_STAGE_IDLE;
    motor->busy = false;
    Stepper_WritePattern(motor, pattern);
}

bool Stepper_SetSequenceMode(Stepper_t *motor, uint8_t sequence_mode)
{
    if (motor->busy || (sequence_mode >= 6U))
    {
        return false;
    }

    motor->sequence_mode = sequence_mode;
    motor->phase = 0U;
    motor->stage = STEPPER_STAGE_IDLE;
    Stepper_Release(motor);
    return true;
}

bool Stepper_IsBusy(const Stepper_t *motor)
{
    return motor->busy;
}
