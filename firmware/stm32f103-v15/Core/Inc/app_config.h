#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "stm32f1xx_hal.h"

/*
 * Hardware mapping
 * Z probe:    ULN2003 INA/INB/INC/IND -> PA0/PA1/PA2/PA3
 * Pump relay: the only fitted pump actuator -> PB6
 * PA4..PA7 are retired outputs held low so a connected old driver stays off.
 * PB7 is deliberately unused.
 *
 * The AdventureX demo only admits three bounded downward probe presets.
 * There is deliberately no automatic upward command: an operator lifts the
 * weak mechanism by hand and then explicitly confirms ZHOME before another
 * descent can be admitted.
 */
#define Z_INA_GPIO_Port GPIOA
#define Z_INA_Pin       GPIO_PIN_0
#define Z_INB_GPIO_Port GPIOA
#define Z_INB_Pin       GPIO_PIN_1
#define Z_INC_GPIO_Port GPIOA
#define Z_INC_Pin       GPIO_PIN_2
#define Z_IND_GPIO_Port GPIOA
#define Z_IND_Pin       GPIO_PIN_3

#define PUMP_GPIO_Port GPIOB
#define PUMP_Pin       GPIO_PIN_6

/*
 * The fitted 5 V relay module is active low and PB6 uses open-drain output:
 * GPIO_PIN_RESET sinks the relay input to ground and energizes the relay.
 * GPIO_PIN_SET releases PB6 (high impedance), allowing the relay module's
 * own input pull-up to turn the relay off.
 *
 * Keep this setting synchronized with the PB6 startup level and open-drain
 * mode in gpio.c and STM32F103_Irrigation.ioc.
 */
#define RELAY_ACTIVE_LEVEL   GPIO_PIN_RESET
#define RELAY_INACTIVE_LEVEL ((RELAY_ACTIVE_LEVEL == GPIO_PIN_SET) ? \
                              GPIO_PIN_RESET : GPIO_PIN_SET)

/* Positive Z steps move upward on the current mechanism. */
#define Z_POSITIVE_DIRECTION  (1)

/*
 * Uncalibrated competition presets.  These are distinct step counts, not
 * centimetre claims.  They must be measured against the final mechanism
 * before the labels "shallow/medium/deep" are used as physical depths.
 */
#define Z_DEPTH_SHALLOW_STEPS       (1024L)
#define Z_DEPTH_MEDIUM_STEPS        (1536L)
#define Z_DEPTH_DEEP_STEPS          (2048L)
#define Z_DEPTH_STEP_INTERVAL_MS    (12U)

/*
 * Loaded 28BYJ-48 profile.
 * Modes 3..5 are dual-phase modes. With INA/INB/INC/IND wired in order to
 * PA0..PA3, mode 3 is the standard A-B-C-D full-step sequence for an
 * original 28BYJ-48 plug and gives the highest normal running torque.
 */
#define Z_DEFAULT_SEQUENCE_MODE       (3U)
#define STEPPER_PRECHARGE_MS          (150U)
#define STEPPER_START_INTERVAL_MS     (50U)
#define STEPPER_RAMP_EVERY_STEPS      (4U)
#define STEPPER_FINAL_HOLD_MS         (300U)

/* Command safety limits. Intervals below 5 ms are rejected for this load. */
#define APP_MIN_STEP_INTERVAL_MS  (5U)
#define APP_MAX_STEP_INTERVAL_MS  (100U)
#define APP_MAX_MOVE_STEPS        (100000L)
#define APP_UART_RX_BUFFER_SIZE   (256U)
#define APP_COMMAND_LINE_SIZE     (96U)

/*
 * STM32F103/PB6 single-pump profile.
 * The binary wire format remains 0xAA 0x55 TYPE LEN PAYLOAD SUM8.
 * This real board has one pump and no HX711 or external safety inputs, so it
 * must not advertise the full F407 capability mask.
 */
#define APP_PROTOCOL_VERSION            (1U)
#define APP_FIRMWARE_BUILD_ID           (2026072515UL)
#define APP_HARDWARE_VARIANT            (2U)
#define APP_FIRMWARE_BUILD_TAG          "rs-f103-z3-pb6-v15"
#define APP_FIRMWARE_VERSION            "2026-07-25-RS-F103-Z3-PB6-V15"
#define APP_CAPABILITY_MASK             (0x00000079UL)

#define APP_HEARTBEAT_TIMEOUT_MS        (1000UL)
#define APP_PERIODIC_REPORT_MS          (200UL)
#define APP_MIN_WATER_DURATION_MS       (100UL)
#define APP_MAX_WATER_DURATION_MS       (30000UL)
#define APP_MIN_TASK_HARD_TIMEOUT_MS    (500UL)
#define APP_MAX_TASK_HARD_TIMEOUT_MS    (120000UL)

/*
 * Competition profile:
 * - arbitrary MOVE/COIL/MODE commands remain locked;
 * - only DEPTH,0..3 presets are admitted after manual ZHOME confirmation;
 * - each ZHOME confirmation admits at most one downward move;
 * - an unbounded ASCII PUMP,1,ON is forbidden;
 * - the only pump-on path is ARM_TIMED_TASK;
 * - heartbeat loss, UART fault, hard timeout, STOP or reset always turns PB6
 *   off, de-energizes PA0..PA3 and latches the controller.
 */
#define APP_ENABLE_MOTION_COMMANDS       (0U)
#define APP_ENABLE_Z_DEPTH_PRESETS       (1U)
#define APP_ALLOW_UNBOUNDED_PUMP_COMMAND (0U)
#define APP_ENABLE_HEARTBEAT_FAILSAFE    (1U)
#define APP_ENABLE_UART_FAULT_FAILSAFE   (1U)

#endif
