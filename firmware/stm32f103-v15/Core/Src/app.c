#include "app.h"
#include "app_config.h"
#include "stepper.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * RootScope real-hardware profile
 * --------------------------------
 * Board: STM32F103C8T6
 * Pump: one relay on PB6 (PB7 is intentionally unused)
 * Motion: one Z-axis 28BYJ-48 + ULN2003, downward presets only
 * Link: 115200-8-N-1, 0xAA 0x55 TYPE LEN PAYLOAD SUM8
 *
 * The host chooses one of three plant-class depth presets.  The MCU admits
 * at most one bounded downward move after an operator confirms manual home.
 * Pump timing remains a separately bounded binary ARM_TIMED_TASK.
 */

enum
{
    CMD_EMERGENCY_STOP = 0x10,
    CMD_CLEAR_ESTOP = 0x11,
    CMD_QUERY_FIRMWARE = 0x12,
    CMD_ARM_MASS_TASK = 0x20,
    CMD_ABORT_TASK = 0x21,
    CMD_ARM_TIMED_TASK = 0x22,
    CMD_HEARTBEAT = 0xFF
};

enum
{
    RSP_FIRMWARE_INFO = 0x81,
    RSP_SAFETY_STATE = 0x82,
    RSP_IRRIGATION_TELEMETRY = 0x83,
    RSP_TASK_RESULT = 0x84,
    RSP_ACK = 0x90,
    RSP_ERROR = 0x9F
};

enum
{
    ACK_OK = 0,
    ACK_REJECTED = 1,
    ACK_LOCKED = 2,
    ACK_BAD_PAYLOAD = 3
};

enum
{
    REASON_NONE = 0,
    REASON_DUPLICATE_SEQ = 1,
    REASON_STALE_SEQ = 2,
    REASON_DUPLICATE_TASK = 3,
    REASON_STALE_TASK = 4,
    REASON_INVALID_CHANNEL = 5,
    REASON_INVALID_TARGET_MASS = 6,
    REASON_INVALID_HARD_TIMEOUT = 7,
    REASON_UNSAFE_INPUT = 8,
    REASON_WATCHDOG_TIMEOUT = 9,
    REASON_BUSY = 10,
    REASON_NO_ACTIVE_TASK = 11,
    REASON_TASK_MISMATCH = 12,
    REASON_CLEAR_CONDITIONS_NOT_MET = 13,
    REASON_HARD_TIMEOUT = 14,
    REASON_MALFORMED_PAYLOAD = 15,
    REASON_UNKNOWN_TYPE = 16,
    REASON_EMERGENCY_STOP = 17,
    REASON_USER_ABORT = 18,
    REASON_BOOT_LOCK = 19,
    REASON_TARGET_REACHED = 20,
    REASON_UNSUPPORTED_CAPABILITY = 21,
    REASON_TIMED_DOSE_COMPLETE = 22,
    REASON_INVALID_DURATION = 23
};

enum
{
    LOCK_NONE = 0,
    LOCK_BOOT = 1,
    LOCK_WATCHDOG = 2,
    LOCK_HARD_TIMEOUT = 3,
    LOCK_EMERGENCY_STOP = 4,
    LOCK_UNSAFE_INPUT = 5,
    LOCK_USER_ABORT = 6
};

enum
{
    TERMINAL_TARGET_REACHED = 1,
    TERMINAL_HARD_TIMEOUT = 2,
    TERMINAL_USER_ABORT = 3,
    TERMINAL_SAFETY_INPUT = 4,
    TERMINAL_EMERGENCY_STOP = 5,
    TERMINAL_WATCHDOG_TIMEOUT = 6,
    TERMINAL_TIMED_DOSE_COMPLETE = 7
};

enum
{
    SAFETY_ESTOP_ACTIVE = 1U << 0,
    SAFETY_WATCHDOG_FRESH = 1U << 5,
    SAFETY_LOCK_LATCHED = 1U << 6,
    SAFETY_ACT_ENABLE = 1U << 7
};

enum
{
    FRAME_WAIT_AA = 0,
    FRAME_WAIT_55,
    FRAME_WAIT_TYPE,
    FRAME_WAIT_LENGTH,
    FRAME_WAIT_PAYLOAD,
    FRAME_WAIT_CHECKSUM
};

#define FRAME_HEADER_AA             (0xAAU)
#define FRAME_HEADER_55             (0x55U)
#define FRAME_MAX_RX_PAYLOAD        (64U)
#define FRAME_MAX_TX_PAYLOAD        (49U)
#define FRAME_MAX_TX_BYTES          (FRAME_MAX_TX_PAYLOAD + 5U)
#define UNKNOWN_MASS_SENTINEL       ((int32_t)(-2147483647L - 1L))

typedef struct
{
    bool active;
    uint32_t task_id;
    uint32_t duration_ms;
    uint32_t hard_timeout_ms;
    uint32_t started_ms;
    uint32_t dose_deadline_ms;
    uint32_t hard_deadline_ms;
    uint32_t first_sample_seq;
} TimedTask_t;

static UART_HandleTypeDef *app_uart;
static Stepper_t z_motor;

static uint8_t uart_rx_byte;
static volatile bool uart_rx_armed;
static volatile uint16_t rx_head;
static volatile uint16_t rx_tail;
static volatile bool rx_overflow;
static uint8_t rx_buffer[APP_UART_RX_BUFFER_SIZE];

static char command_line[APP_COMMAND_LINE_SIZE];
static uint16_t command_length;

static uint8_t frame_state;
static uint8_t frame_type;
static uint8_t frame_length;
static uint8_t frame_position;
static uint8_t frame_sum;
static uint8_t frame_payload[FRAME_MAX_RX_PAYLOAD];
static bool binary_session_seen;

static bool pump_on;
static bool z_was_busy;
static bool z_home_confirmed;
static bool z_descent_committed;
static uint8_t z_selected_depth;
static uint32_t z_selected_steps;
static bool lock_latched;
static uint8_t lock_reason;
static uint16_t blocked_count;
static bool heartbeat_seen;
static uint32_t last_heartbeat_ms;
static bool sequence_seen;
static uint16_t last_sequence;
static uint32_t last_task_id;
static uint32_t telemetry_sequence;
static uint16_t task_result_sequence;
static uint32_t boot_id_low;
static uint32_t boot_id_high;
static uint32_t next_periodic_report_ms;
static TimedTask_t timed_task;

static void ArmUartReceive(void)
{
    HAL_StatusTypeDef status;

    if ((app_uart == NULL) || uart_rx_armed)
    {
        return;
    }

    status = HAL_UART_Receive_IT(app_uart, &uart_rx_byte, 1U);
    /*
     * HAL_BUSY does not prove that this byte buffer was armed. In particular,
     * an error callback can run while HAL is still unwinding the old request.
     * Leave the flag clear so App_Task retries instead of losing UART forever.
     */
    if (status == HAL_OK)
    {
        uart_rx_armed = true;
    }
}

static bool TimeReached(uint32_t now_ms, uint32_t deadline_ms)
{
    return ((int32_t)(now_ms - deadline_ms) >= 0);
}

static void Reply(const char *format, ...)
{
    char message[160];
    va_list args;
    int length;

    va_start(args, format);
    length = vsnprintf(message, sizeof(message), format, args);
    va_end(args);

    if ((length <= 0) || (app_uart == NULL))
    {
        return;
    }
    if ((size_t)length >= sizeof(message))
    {
        length = (int)sizeof(message) - 1;
    }
    (void)HAL_UART_Transmit(app_uart, (uint8_t *)message,
                            (uint16_t)length, 100U);
}

static uint16_t ReadU16(const uint8_t *data)
{
    return (uint16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8U));
}

static uint32_t ReadU32(const uint8_t *data)
{
    return (uint32_t)data[0] |
           ((uint32_t)data[1] << 8U) |
           ((uint32_t)data[2] << 16U) |
           ((uint32_t)data[3] << 24U);
}

static void WriteU16(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8U) & 0xFFU);
}

static void WriteU32(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8U) & 0xFFU);
    data[2] = (uint8_t)((value >> 16U) & 0xFFU);
    data[3] = (uint8_t)((value >> 24U) & 0xFFU);
}

static void WriteI32(uint8_t *data, int32_t value)
{
    WriteU32(data, (uint32_t)value);
}

static void WriteU64Parts(uint8_t *data, uint32_t low, uint32_t high)
{
    WriteU32(data, low);
    WriteU32(data + 4U, high);
}

static void SendFrame(uint8_t type, const uint8_t *payload, uint8_t length)
{
    uint8_t frame[FRAME_MAX_TX_BYTES];
    uint8_t checksum;
    uint8_t i;

    if ((app_uart == NULL) || (length > FRAME_MAX_TX_PAYLOAD))
    {
        return;
    }

    frame[0] = FRAME_HEADER_AA;
    frame[1] = FRAME_HEADER_55;
    frame[2] = type;
    frame[3] = length;
    checksum = (uint8_t)(FRAME_HEADER_AA + FRAME_HEADER_55 + type + length);
    for (i = 0U; i < length; ++i)
    {
        frame[4U + i] = payload[i];
        checksum = (uint8_t)(checksum + payload[i]);
    }
    frame[4U + length] = checksum;
    (void)HAL_UART_Transmit(app_uart, frame, (uint16_t)(length + 5U), 100U);
}

static void IncrementBlockedCount(void)
{
    if (blocked_count != 0xFFFFU)
    {
        ++blocked_count;
    }
}

static uint32_t HeartbeatAgeMs(uint32_t now_ms)
{
    return heartbeat_seen ? (now_ms - last_heartbeat_ms) : 0xFFFFFFFFUL;
}

static bool HeartbeatFresh(uint32_t now_ms)
{
    return heartbeat_seen &&
           (HeartbeatAgeMs(now_ms) <= APP_HEARTBEAT_TIMEOUT_MS);
}

static uint16_t CurrentSafetyBits(uint32_t now_ms)
{
    uint16_t bits = 0U;

    if (lock_reason == LOCK_EMERGENCY_STOP)
    {
        bits |= SAFETY_ESTOP_ACTIVE;
    }
    if (HeartbeatFresh(now_ms))
    {
        bits |= SAFETY_WATCHDOG_FRESH;
    }
    if (lock_latched)
    {
        bits |= SAFETY_LOCK_LATCHED;
    }
    if (!lock_latched && HeartbeatFresh(now_ms))
    {
        bits |= SAFETY_ACT_ENABLE;
    }
    return bits;
}

static void Pump_WriteOutput(bool enable)
{
    GPIO_PinState output_level =
        enable ? RELAY_ACTIVE_LEVEL : RELAY_INACTIVE_LEVEL;

    HAL_GPIO_WritePin(PUMP_GPIO_Port, PUMP_Pin, output_level);
    pump_on = enable;
}

static void Pump_OutputOff(void)
{
    /* Release the relay so COM/NO opens and the pump loses power. */
    Pump_WriteOutput(false);
}

static void Pump_OutputOn(void)
{
    /*
     * Energize the relay so COM/NO closes and applies 5 V to the pump.
     * There is no timer or automatic watering path in this firmware profile.
     */
    Pump_WriteOutput(true);
}

static void SendIoStatus(void)
{
    uint32_t pa_odr = GPIOA->ODR & 0xFFUL;
    uint32_t pb6_odr =
        ((PUMP_GPIO_Port->ODR & (uint32_t)PUMP_Pin) != 0U) ? 1UL : 0UL;

    /*
     * This is a diagnostic view of the MCU output latches. It deliberately
     * leaves every output unchanged.  Raw COIL is intentionally disabled in
     * the competition build.
     */
    Reply("IOSTATUS,PA=0x%02lX,Z=0x%lX,"
          "PB6=%lu,PLOG=%u\r\n",
          (unsigned long)pa_odr,
          (unsigned long)(pa_odr & 0x0FUL),
          (unsigned long)pb6_odr,
          pump_on ? 1U : 0U);
}

static void StopAllActuators(void)
{
    Stepper_Stop(&z_motor);
    Pump_OutputOff();
    z_was_busy = false;
}

static void SendAck(uint8_t command_type, uint16_t sequence,
                    uint8_t status, uint8_t reason, uint32_t task_id)
{
    uint8_t payload[9];

    payload[0] = command_type;
    WriteU16(payload + 1U, sequence);
    payload[3] = status;
    payload[4] = reason;
    WriteU32(payload + 5U, task_id);
    SendFrame(RSP_ACK, payload, (uint8_t)sizeof(payload));
}

static void SendError(uint8_t reason)
{
    SendFrame(RSP_ERROR, &reason, 1U);
}

static void SendFirmwareInfo(void)
{
    uint8_t payload[35];
    const char *tag = APP_FIRMWARE_BUILD_TAG;
    size_t tag_length = strlen(tag);

    memset(payload, 0, sizeof(payload));
    WriteU16(payload, APP_PROTOCOL_VERSION);
    WriteU32(payload + 2U, APP_CAPABILITY_MASK);
    WriteU32(payload + 6U, APP_FIRMWARE_BUILD_ID);
    payload[10] = APP_HARDWARE_VARIANT;
    if (tag_length > 16U)
    {
        tag_length = 16U;
    }
    memcpy(payload + 11U, tag, tag_length);
    WriteU64Parts(payload + 27U, boot_id_low, boot_id_high);
    SendFrame(RSP_FIRMWARE_INFO, payload, (uint8_t)sizeof(payload));
}

static void SendSafetyState(uint32_t now_ms)
{
    uint8_t payload[17];

    WriteU64Parts(payload, boot_id_low, boot_id_high);
    WriteU16(payload + 8U, CurrentSafetyBits(now_ms));
    WriteU16(payload + 10U, blocked_count);
    payload[12] = lock_reason;
    WriteU32(payload + 13U, HeartbeatAgeMs(now_ms));
    SendFrame(RSP_SAFETY_STATE, payload, (uint8_t)sizeof(payload));
}

static void SendTelemetry(uint32_t now_ms)
{
    uint8_t payload[23];
    uint32_t task_id = timed_task.active ? timed_task.task_id : 0U;

    ++telemetry_sequence;
    if (telemetry_sequence == 0U)
    {
        ++telemetry_sequence;
    }
    WriteU32(payload, task_id);
    WriteU32(payload + 4U, telemetry_sequence);
    payload[8] = pump_on ? 0x01U : 0x00U;
    WriteI32(payload + 9U, UNKNOWN_MASS_SENTINEL);
    WriteI32(payload + 13U, UNKNOWN_MASS_SENTINEL);
    WriteU16(payload + 17U, CurrentSafetyBits(now_ms));
    WriteU32(payload + 19U, now_ms);
    SendFrame(RSP_IRRIGATION_TELEMETRY, payload, (uint8_t)sizeof(payload));
}

static void SendTaskResult(uint32_t task_id, uint8_t terminal_reason,
                           uint32_t first_sample_seq, uint32_t now_ms)
{
    uint8_t payload[49];
    uint32_t final_sample_seq = telemetry_sequence;

    if (first_sample_seq == 0U)
    {
        first_sample_seq = 1U;
    }
    if (final_sample_seq < first_sample_seq)
    {
        final_sample_seq = first_sample_seq;
    }
    ++task_result_sequence;
    if (task_result_sequence == 0U)
    {
        ++task_result_sequence;
    }

    memset(payload, 0, sizeof(payload));
    WriteU64Parts(payload, boot_id_low, boot_id_high);
    WriteU32(payload + 8U, task_id);
    WriteU16(payload + 12U, task_result_sequence);
    payload[14] = terminal_reason;
    /* Bytes 15..22 are zero: no physical mass sensor is fitted. */
    WriteU32(payload + 23U, first_sample_seq);
    WriteU32(payload + 27U, final_sample_seq);
    WriteU16(payload + 31U, 1U);
    /* Bytes 33..40 are zero and scale_stable (byte 41) remains false. */
    WriteU32(payload + 42U, now_ms);
    payload[46] = 0U;
    WriteU16(payload + 47U, CurrentSafetyBits(now_ms));
    SendFrame(RSP_TASK_RESULT, payload, (uint8_t)sizeof(payload));
}

static void FinishTimedTask(uint8_t terminal_reason, bool latch,
                            uint8_t new_lock_reason, uint32_t now_ms)
{
    uint32_t task_id;
    uint32_t first_sample_seq;

    if (!timed_task.active)
    {
        return;
    }
    task_id = timed_task.task_id;
    first_sample_seq = timed_task.first_sample_seq;
    Pump_OutputOff();
    memset(&timed_task, 0, sizeof(timed_task));
    if (latch)
    {
        lock_latched = true;
        lock_reason = new_lock_reason;
        StopAllActuators();
        IncrementBlockedCount();
    }
    SendTaskResult(task_id, terminal_reason, first_sample_seq, now_ms);
}

static void LatchEmergency(uint8_t new_lock_reason, uint8_t terminal_reason,
                           uint32_t now_ms)
{
    bool had_task = timed_task.active;
    uint32_t task_id = timed_task.task_id;
    uint32_t first_sample_seq = timed_task.first_sample_seq;

    StopAllActuators();
    memset(&timed_task, 0, sizeof(timed_task));
    lock_latched = true;
    lock_reason = new_lock_reason;
    IncrementBlockedCount();
    if (had_task)
    {
        SendTaskResult(task_id, terminal_reason, first_sample_seq, now_ms);
    }
}

static uint8_t CheckSequence(uint16_t sequence)
{
    uint16_t delta;

    if (sequence == 0U)
    {
        return REASON_STALE_SEQ;
    }
    if (!sequence_seen)
    {
        return REASON_NONE;
    }
    delta = (uint16_t)(sequence - last_sequence);
    if (delta == 0U)
    {
        return REASON_DUPLICATE_SEQ;
    }
    if (delta > 32767U)
    {
        return REASON_STALE_SEQ;
    }
    return REASON_NONE;
}

static void CommitSequence(uint16_t sequence)
{
    if (sequence != 0U)
    {
        last_sequence = sequence;
        sequence_seen = true;
    }
}

static void RejectSequence(uint8_t type, uint16_t sequence, uint8_t reason,
                           uint32_t task_id)
{
    IncrementBlockedCount();
    SendAck(type, sequence, ACK_REJECTED, reason, task_id);
}

static void HandleBinaryFrame(uint8_t type, const uint8_t *payload,
                              uint8_t length, uint32_t now_ms)
{
    uint16_t sequence = 0U;
    uint8_t sequence_issue;
    uint32_t task_id = 0U;

    binary_session_seen = true;

    if (type == CMD_EMERGENCY_STOP)
    {
        if (length != 2U)
        {
            IncrementBlockedCount();
            SendAck(type, 0U, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, 0U);
            return;
        }
        sequence = ReadU16(payload);
        if (CheckSequence(sequence) == REASON_NONE)
        {
            CommitSequence(sequence);
        }
        LatchEmergency(LOCK_EMERGENCY_STOP,
                       TERMINAL_EMERGENCY_STOP, now_ms);
        SendAck(type, sequence, ACK_OK, REASON_EMERGENCY_STOP, 0U);
        return;
    }

    if ((type == CMD_ARM_MASS_TASK) || (type == CMD_ARM_TIMED_TASK) ||
        (type == CMD_ABORT_TASK))
    {
        if (length >= 6U)
        {
            task_id = ReadU32(payload);
            sequence = ReadU16(payload + 4U);
        }
    }
    else if (length >= 2U)
    {
        sequence = ReadU16(payload);
    }

    sequence_issue = CheckSequence(sequence);
    if (sequence_issue != REASON_NONE)
    {
        RejectSequence(type, sequence, sequence_issue, task_id);
        return;
    }

    if (type == CMD_HEARTBEAT)
    {
        if (length != 3U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, 0U);
            return;
        }
        CommitSequence(sequence);
        heartbeat_seen = true;
        last_heartbeat_ms = now_ms;
        SendAck(type, sequence, ACK_OK, REASON_NONE, 0U);
        return;
    }

    if (type == CMD_QUERY_FIRMWARE)
    {
        if (length != 2U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, 0U);
            return;
        }
        CommitSequence(sequence);
        SendFirmwareInfo();
        SendAck(type, sequence, ACK_OK, REASON_NONE, 0U);
        return;
    }

    if (type == CMD_CLEAR_ESTOP)
    {
        if (length != 2U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, 0U);
            return;
        }
        CommitSequence(sequence);
        if (!HeartbeatFresh(now_ms) || pump_on || timed_task.active ||
            Stepper_IsBusy(&z_motor))
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_LOCKED,
                    REASON_CLEAR_CONDITIONS_NOT_MET, 0U);
            return;
        }
        lock_latched = false;
        lock_reason = LOCK_NONE;
        SendAck(type, sequence, ACK_OK, REASON_NONE, 0U);
        return;
    }

    if (type == CMD_ARM_MASS_TASK)
    {
        if (length != 23U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, task_id);
            return;
        }
        CommitSequence(sequence);
        IncrementBlockedCount();
        SendAck(type, sequence, ACK_REJECTED,
                REASON_UNSUPPORTED_CAPABILITY, task_id);
        return;
    }

    if (type == CMD_ARM_TIMED_TASK)
    {
        uint8_t channel;
        uint32_t duration_ms;
        uint32_t hard_timeout_ms;

        if (length != 23U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, task_id);
            return;
        }
        channel = payload[6];
        duration_ms = ReadU32(payload + 7U);
        hard_timeout_ms = ReadU32(payload + 11U);
        CommitSequence(sequence);

        if (lock_latched || !HeartbeatFresh(now_ms))
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_LOCKED,
                    HeartbeatFresh(now_ms) ?
                    REASON_BOOT_LOCK : REASON_WATCHDOG_TIMEOUT,
                    task_id);
            return;
        }
        if (timed_task.active || pump_on || Stepper_IsBusy(&z_motor))
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_BUSY, task_id);
            return;
        }
        if (task_id == 0U || task_id < last_task_id)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_STALE_TASK, task_id);
            return;
        }
        if (task_id == last_task_id)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_DUPLICATE_TASK, task_id);
            return;
        }
        if (channel != 1U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_INVALID_CHANNEL, task_id);
            return;
        }
        if ((duration_ms < APP_MIN_WATER_DURATION_MS) ||
            (duration_ms > APP_MAX_WATER_DURATION_MS))
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_INVALID_DURATION, task_id);
            return;
        }
        if ((hard_timeout_ms < APP_MIN_TASK_HARD_TIMEOUT_MS) ||
            (hard_timeout_ms > APP_MAX_TASK_HARD_TIMEOUT_MS) ||
            (hard_timeout_ms < duration_ms))
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_INVALID_HARD_TIMEOUT, task_id);
            return;
        }

        last_task_id = task_id;
        timed_task.active = true;
        timed_task.task_id = task_id;
        timed_task.duration_ms = duration_ms;
        timed_task.hard_timeout_ms = hard_timeout_ms;
        timed_task.started_ms = now_ms;
        timed_task.dose_deadline_ms = now_ms + duration_ms;
        timed_task.hard_deadline_ms = now_ms + hard_timeout_ms;
        timed_task.first_sample_seq =
            (telemetry_sequence == 0U) ? 1U : telemetry_sequence;
        Pump_OutputOn();
        SendAck(type, sequence, ACK_OK, REASON_NONE, task_id);
        return;
    }

    if (type == CMD_ABORT_TASK)
    {
        if (length != 7U)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_BAD_PAYLOAD,
                    REASON_MALFORMED_PAYLOAD, task_id);
            return;
        }
        CommitSequence(sequence);
        if (!timed_task.active)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_NO_ACTIVE_TASK, task_id);
            return;
        }
        if (task_id != timed_task.task_id)
        {
            IncrementBlockedCount();
            SendAck(type, sequence, ACK_REJECTED,
                    REASON_TASK_MISMATCH, task_id);
            return;
        }
        SendAck(type, sequence, ACK_OK, REASON_USER_ABORT, task_id);
        FinishTimedTask(TERMINAL_USER_ABORT, true, LOCK_USER_ABORT, now_ms);
        return;
    }

    IncrementBlockedCount();
    SendError(REASON_UNKNOWN_TYPE);
}

static bool ParseLong(const char *text, int32_t *value)
{
    char *end;
    long parsed;

    if ((text == NULL) || (*text == '\0'))
    {
        return false;
    }
    parsed = strtol(text, &end, 10);
    if ((*end != '\0') || (parsed < INT32_MIN) || (parsed > INT32_MAX))
    {
        return false;
    }
    *value = (int32_t)parsed;
    return true;
}

static bool ParseU32(const char *text, uint32_t *value)
{
    int32_t parsed;

    if (!ParseLong(text, &parsed) || (parsed < 0))
    {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static bool ValidStepSettings(int32_t steps, uint32_t interval_ms)
{
    int64_t magnitude = (steps < 0) ? -(int64_t)steps : (int64_t)steps;
    return (steps != 0) &&
           (magnitude <= APP_MAX_MOVE_STEPS) &&
           (interval_ms >= APP_MIN_STEP_INTERVAL_MS) &&
           (interval_ms <= APP_MAX_STEP_INTERVAL_MS);
}

static uint32_t DepthPresetSteps(uint32_t depth_level)
{
    if (depth_level == 1U)
    {
        return (uint32_t)Z_DEPTH_SHALLOW_STEPS;
    }
    if (depth_level == 2U)
    {
        return (uint32_t)Z_DEPTH_MEDIUM_STEPS;
    }
    if (depth_level == 3U)
    {
        return (uint32_t)Z_DEPTH_DEEP_STEPS;
    }
    return 0U;
}

static void HandleZHome(char *confirmation_text)
{
#if !APP_ENABLE_Z_DEPTH_PRESETS
    (void)confirmation_text;
    Reply("ERR,Z_DEPTH_PRESETS_LOCKED\r\n");
    return;
#endif

    if ((confirmation_text == NULL) ||
        (strcmp(confirmation_text, "CONFIRM") != 0))
    {
        Reply("ERR,ZHOME_REQUIRES_CONFIRM\r\n");
        return;
    }
    if (Stepper_IsBusy(&z_motor) || pump_on || timed_task.active)
    {
        Reply("ERR,ACTUATOR_BUSY\r\n");
        return;
    }

    /*
     * This command never moves the mechanism.  It records the operator's
     * statement that the probe was physically returned to the top by hand.
     */
    Stepper_Stop(&z_motor);
    z_home_confirmed = true;
    z_descent_committed = false;
    z_selected_depth = 0U;
    z_selected_steps = 0U;
    Reply("ACK,ZHOME,CONFIRMED,MANUAL_OBSERVATION\r\n");
}

static void HandleDepthPreset(char *depth_text, uint32_t now_ms)
{
    uint32_t depth_level;
    uint32_t steps;
    int32_t signed_steps;

#if !APP_ENABLE_Z_DEPTH_PRESETS
    (void)depth_text;
    (void)now_ms;
    Reply("ERR,Z_DEPTH_PRESETS_LOCKED\r\n");
    return;
#endif

    if (!ParseU32(depth_text, &depth_level) || (depth_level > 3U))
    {
        Reply("ERR,BAD_DEPTH_PRESET\r\n");
        return;
    }
    if (depth_level == 0U)
    {
        Reply("ACK,DEPTH,0,HOLD,NO_MOTION\r\n");
        return;
    }
    if (pump_on || timed_task.active)
    {
        Reply("ERR,PUMP_ACTIVE\r\n");
        return;
    }
    if (Stepper_IsBusy(&z_motor))
    {
        Reply("ERR,AXIS_BUSY\r\n");
        return;
    }
    if (lock_latched || !HeartbeatFresh(now_ms))
    {
        Reply("ERR,SAFETY_LOCKED\r\n");
        return;
    }
    if (!z_home_confirmed)
    {
        Reply("ERR,ZHOME_NOT_CONFIRMED\r\n");
        return;
    }
    if (z_descent_committed)
    {
        Reply("ERR,Z_DESCENT_ALREADY_USED\r\n");
        return;
    }

    steps = DepthPresetSteps(depth_level);
    if ((steps == 0U) || (steps > (uint32_t)APP_MAX_MOVE_STEPS))
    {
        Reply("ERR,BAD_DEPTH_CONFIG\r\n");
        return;
    }
    signed_steps = -(int32_t)steps * Z_POSITIVE_DIRECTION;
    if (!Stepper_Move(&z_motor, signed_steps,
                      Z_DEPTH_STEP_INTERVAL_MS, now_ms))
    {
        Reply("ERR,AXIS_BUSY\r\n");
        return;
    }

    /*
     * Commit before reporting ACK.  STOP/watchdog interruptions do not clear
     * this latch; only a new physical manual-home confirmation can do so.
     */
    z_home_confirmed = false;
    z_descent_committed = true;
    z_selected_depth = (uint8_t)depth_level;
    z_selected_steps = steps;
    Reply("ACK,DEPTH,%lu,DOWN,STEPS=%lu,MANUAL_RETURN_REQUIRED\r\n",
          (unsigned long)depth_level, (unsigned long)steps);
}

static void HandleMove(char *axis_text, char *steps_text, char *interval_text,
                       uint32_t now_ms)
{
    int32_t steps;
    uint32_t interval_ms;
    int32_t calibrated_steps;

#if !APP_ENABLE_MOTION_COMMANDS
    (void)axis_text;
    (void)steps_text;
    (void)interval_text;
    (void)now_ms;
    Reply("ERR,MOTION_EXTENSION_LOCKED\r\n");
    return;
#endif

    if (pump_on || timed_task.active)
    {
        Reply("ERR,PUMP_ACTIVE\r\n");
        return;
    }
    if ((axis_text == NULL) || (steps_text == NULL) ||
        (interval_text == NULL) || !ParseLong(steps_text, &steps) ||
        !ParseU32(interval_text, &interval_ms) ||
        !ValidStepSettings(steps, interval_ms))
    {
        Reply("ERR,BAD_MOVE\r\n");
        return;
    }
    if (strcmp(axis_text, "Z") != 0)
    {
        Reply("ERR,BAD_AXIS\r\n");
        return;
    }
    calibrated_steps = steps * Z_POSITIVE_DIRECTION;

    if (Stepper_IsBusy(&z_motor) ||
        !Stepper_Move(&z_motor, calibrated_steps, interval_ms, now_ms))
    {
        Reply("ERR,AXIS_BUSY\r\n");
        return;
    }
    Reply("ACK,MOVE,%s\r\n", axis_text);
}

static void HandlePump(char *id_text, char *state_text)
{
    uint32_t pump_id;

    if (!ParseU32(id_text, &pump_id) || (pump_id != 1U) ||
        (state_text == NULL))
    {
        Reply("ERR,BAD_PUMP\r\n");
        return;
    }
    if (strcmp(state_text, "OFF") == 0)
    {
        Pump_OutputOff();
        Reply("ACK,PUMP,1,OFF\r\n");
        return;
    }
    if (strcmp(state_text, "ON") == 0)
    {
#if APP_ALLOW_UNBOUNDED_PUMP_COMMAND
        Pump_OutputOn();
        Reply("ACK,PUMP,1,ON\r\n");
#else
        Reply("ERR,UNBOUNDED_PUMP_DISABLED\r\n");
#endif
        return;
    }
    Reply("ERR,BAD_PUMP_STATE\r\n");
}

static void HandleCoil(char *axis_text, char *pattern_text)
{
    uint32_t pattern;

#if !APP_ENABLE_MOTION_COMMANDS
    (void)axis_text;
    (void)pattern_text;
    Reply("ERR,MOTION_EXTENSION_LOCKED\r\n");
    return;
#endif

    if (pump_on || timed_task.active)
    {
        Reply("ERR,PUMP_ACTIVE\r\n");
        return;
    }
    if ((axis_text == NULL) || (pattern_text == NULL) ||
        !ParseU32(pattern_text, &pattern) || (pattern > 15U))
    {
        Reply("ERR,BAD_COIL\r\n");
        return;
    }
    if (strcmp(axis_text, "Z") != 0)
    {
        Reply("ERR,BAD_AXIS\r\n");
        return;
    }
    if (Stepper_IsBusy(&z_motor))
    {
        Reply("ERR,AXIS_BUSY\r\n");
        return;
    }
    Stepper_SetRawPattern(&z_motor, (uint8_t)pattern);
    Reply("ACK,COIL,%s,%lu\r\n", axis_text, (unsigned long)pattern);
}

static void HandleMode(char *axis_text, char *mode_text)
{
    uint32_t mode;

#if !APP_ENABLE_MOTION_COMMANDS
    (void)axis_text;
    (void)mode_text;
    Reply("ERR,MOTION_EXTENSION_LOCKED\r\n");
    return;
#endif

    if ((axis_text == NULL) || (mode_text == NULL) ||
        !ParseU32(mode_text, &mode) || (mode > 5U))
    {
        Reply("ERR,BAD_MODE\r\n");
        return;
    }
    if (strcmp(axis_text, "Z") != 0)
    {
        Reply("ERR,BAD_AXIS\r\n");
        return;
    }
    if (!Stepper_SetSequenceMode(&z_motor, (uint8_t)mode))
    {
        Reply("ERR,AXIS_BUSY\r\n");
        return;
    }
    Reply("ACK,MODE,%s,%lu\r\n", axis_text, (unsigned long)mode);
}

static void HandleAsciiCommand(char *line, uint32_t now_ms)
{
    char *argv[6] = {0};
    char *token;
    uint8_t argc = 0U;

    token = strtok(line, ",");
    while ((token != NULL) && (argc < 6U))
    {
        argv[argc++] = token;
        token = strtok(NULL, ",");
    }

    if ((argc == 1U) && (strcmp(argv[0], "PING") == 0))
    {
        Reply("ACK,PONG\r\n");
    }
    else if ((argc == 1U) && (strcmp(argv[0], "VERSION") == 0))
    {
        Reply("VERSION,%s,HW=F103C8,PUMP=PB6,RELAY=ACTIVE_LOW_OD,"
              "MOTION=Z3_DOWN_ONLY,RETURN=MANUAL,CAPS=0x%08lX\r\n",
              APP_FIRMWARE_VERSION, (unsigned long)APP_CAPABILITY_MASK);
    }
    else if ((argc == 1U) && (strcmp(argv[0], "STATUS") == 0))
    {
        Reply("STATUS,Z=%u,P=%u,LOCK=%u,REASON=%u,HB=%u,TASK=%lu,"
              "ZHOME=%u,ZUSED=%u,ZLEVEL=%u,ZSTEPS=%lu\r\n",
              Stepper_IsBusy(&z_motor) ? 1U : 0U,
              pump_on ? 1U : 0U,
              lock_latched ? 1U : 0U,
              (unsigned int)lock_reason,
              HeartbeatFresh(now_ms) ? 1U : 0U,
              (unsigned long)(timed_task.active ? timed_task.task_id : 0U),
              z_home_confirmed ? 1U : 0U,
              z_descent_committed ? 1U : 0U,
              (unsigned int)z_selected_depth,
              (unsigned long)z_selected_steps);
    }
    else if ((argc == 1U) && (strcmp(argv[0], "IOSTATUS") == 0))
    {
        SendIoStatus();
    }
    else if ((argc == 1U) && (strcmp(argv[0], "STOP") == 0))
    {
        LatchEmergency(LOCK_EMERGENCY_STOP,
                       TERMINAL_EMERGENCY_STOP, now_ms);
        Reply("ACK,STOP,LOCKED\r\n");
    }
    else if ((argc == 4U) && (strcmp(argv[0], "MOVE") == 0))
    {
        HandleMove(argv[1], argv[2], argv[3], now_ms);
    }
    else if ((argc == 2U) && (strcmp(argv[0], "ZHOME") == 0))
    {
        HandleZHome(argv[1]);
    }
    else if ((argc == 2U) && (strcmp(argv[0], "DEPTH") == 0))
    {
        HandleDepthPreset(argv[1], now_ms);
    }
    else if ((argc == 3U) && (strcmp(argv[0], "PUMP") == 0))
    {
        HandlePump(argv[1], argv[2]);
    }
    else if ((argc == 3U) && (strcmp(argv[0], "COIL") == 0))
    {
        HandleCoil(argv[1], argv[2]);
    }
    else if ((argc == 3U) && (strcmp(argv[0], "MODE") == 0))
    {
        HandleMode(argv[1], argv[2]);
    }
    else
    {
        Reply("ERR,UNKNOWN_COMMAND\r\n");
    }
}

static void ResetFrameParser(void)
{
    frame_state = FRAME_WAIT_AA;
    frame_type = 0U;
    frame_length = 0U;
    frame_position = 0U;
    frame_sum = 0U;
}

static void ProcessInputByte(uint8_t byte, uint32_t now_ms)
{
    if (frame_state == FRAME_WAIT_AA)
    {
        if (byte == FRAME_HEADER_AA)
        {
            frame_state = FRAME_WAIT_55;
            frame_sum = FRAME_HEADER_AA;
            return;
        }
        if ((byte == '\r') || (byte == '\n'))
        {
            if (command_length > 0U)
            {
                command_line[command_length] = '\0';
                HandleAsciiCommand(command_line, now_ms);
                command_length = 0U;
            }
        }
        else if ((byte >= 0x20U) && (byte <= 0x7EU))
        {
            if (command_length < (APP_COMMAND_LINE_SIZE - 1U))
            {
                command_line[command_length++] = (char)byte;
            }
            else
            {
                command_length = 0U;
                Reply("ERR,LINE_TOO_LONG\r\n");
            }
        }
        return;
    }

    if (frame_state == FRAME_WAIT_55)
    {
        if (byte == FRAME_HEADER_55)
        {
            frame_sum = (uint8_t)(frame_sum + byte);
            frame_state = FRAME_WAIT_TYPE;
        }
        else if (byte == FRAME_HEADER_AA)
        {
            frame_sum = FRAME_HEADER_AA;
        }
        else
        {
            ResetFrameParser();
        }
        return;
    }

    if (frame_state == FRAME_WAIT_TYPE)
    {
        frame_type = byte;
        frame_sum = (uint8_t)(frame_sum + byte);
        frame_state = FRAME_WAIT_LENGTH;
        return;
    }

    if (frame_state == FRAME_WAIT_LENGTH)
    {
        frame_length = byte;
        frame_sum = (uint8_t)(frame_sum + byte);
        frame_position = 0U;
        if (frame_length > FRAME_MAX_RX_PAYLOAD)
        {
            IncrementBlockedCount();
            ResetFrameParser();
        }
        else
        {
            frame_state = (frame_length == 0U) ?
                          FRAME_WAIT_CHECKSUM : FRAME_WAIT_PAYLOAD;
        }
        return;
    }

    if (frame_state == FRAME_WAIT_PAYLOAD)
    {
        frame_payload[frame_position++] = byte;
        frame_sum = (uint8_t)(frame_sum + byte);
        if (frame_position >= frame_length)
        {
            frame_state = FRAME_WAIT_CHECKSUM;
        }
        return;
    }

    if (frame_state == FRAME_WAIT_CHECKSUM)
    {
        if (byte == frame_sum)
        {
            HandleBinaryFrame(frame_type, frame_payload,
                              frame_length, now_ms);
        }
        else
        {
            IncrementBlockedCount();
        }
        ResetFrameParser();
    }
}

static void ProcessRx(uint32_t now_ms)
{
    while (rx_tail != rx_head)
    {
        uint8_t byte = rx_buffer[rx_tail];
        rx_tail = (uint16_t)((rx_tail + 1U) % APP_UART_RX_BUFFER_SIZE);
        ProcessInputByte(byte, now_ms);
    }

    if (rx_overflow)
    {
        rx_overflow = false;
        command_length = 0U;
        ResetFrameParser();
        IncrementBlockedCount();
        Reply("ERR,RX_OVERFLOW\r\n");
    }
}

static void ReportAxisCompletion(void)
{
    bool z_busy = Stepper_IsBusy(&z_motor);

    if (z_was_busy && !z_busy)
    {
        Reply("DONE,Z,DEPTH=%u,STEPS=%lu,MANUAL_RETURN_REQUIRED\r\n",
              (unsigned int)z_selected_depth,
              (unsigned long)z_selected_steps);
    }
    z_was_busy = z_busy;
}

static void InitializeBootId(void)
{
#if defined(STM32F103xB)
    uint32_t uid0 = HAL_GetUIDw0();
    uint32_t uid1 = HAL_GetUIDw1();
    uint32_t uid2 = HAL_GetUIDw2();
    uint32_t timing_nonce = SysTick->VAL ^ RCC->CSR ^ HAL_GetTick();

    boot_id_low = uid0 ^ uid2 ^ timing_nonce;
    boot_id_high = uid1 ^ APP_FIRMWARE_BUILD_ID;
#else
    /* Deterministic host-test token; never used as a physical identity claim. */
    boot_id_low = 0xF1030001UL;
    boot_id_high = APP_FIRMWARE_BUILD_ID;
#endif
    if ((boot_id_low == 0U) && (boot_id_high == 0U))
    {
        boot_id_low = 1U;
    }
}

void App_Init(UART_HandleTypeDef *command_uart)
{
    app_uart = command_uart;
    uart_rx_armed = false;
    rx_head = 0U;
    rx_tail = 0U;
    rx_overflow = false;
    command_length = 0U;
    binary_session_seen = false;
    pump_on = false;
    z_was_busy = false;
    z_home_confirmed = false;
    z_descent_committed = false;
    z_selected_depth = 0U;
    z_selected_steps = 0U;
    lock_latched = true;
    lock_reason = LOCK_BOOT;
    blocked_count = 0U;
    heartbeat_seen = false;
    last_heartbeat_ms = 0U;
    sequence_seen = false;
    last_sequence = 0U;
    last_task_id = 0U;
    telemetry_sequence = 0U;
    task_result_sequence = 0U;
    memset(&timed_task, 0, sizeof(timed_task));
    ResetFrameParser();
    InitializeBootId();
    next_periodic_report_ms = HAL_GetTick() + APP_PERIODIC_REPORT_MS;

    Stepper_Init(&z_motor,
                 Z_INA_GPIO_Port, Z_INA_Pin, Z_INB_GPIO_Port, Z_INB_Pin,
                 Z_INC_GPIO_Port, Z_INC_Pin, Z_IND_GPIO_Port, Z_IND_Pin, 1);
    (void)Stepper_SetSequenceMode(&z_motor, Z_DEFAULT_SEQUENCE_MODE);
    StopAllActuators();
    ArmUartReceive();
    Reply("READY,STM32_IRRIGATION,%s,PB6_ACTIVE_LOW_OD,"
          "Z3_DOWN_ONLY,MANUAL_RETURN,BOOT_LOCKED,OUTPUTS_OFF\r\n",
          APP_FIRMWARE_VERSION);
}

void App_Task(void)
{
    uint32_t now_ms = HAL_GetTick();

    ArmUartReceive();
    ProcessRx(now_ms);

#if APP_ENABLE_HEARTBEAT_FAILSAFE
    if (!lock_latched && !HeartbeatFresh(now_ms))
    {
        LatchEmergency(LOCK_WATCHDOG,
                       TERMINAL_WATCHDOG_TIMEOUT, now_ms);
    }
#endif

    if (timed_task.active)
    {
        if (TimeReached(now_ms, timed_task.hard_deadline_ms))
        {
            FinishTimedTask(TERMINAL_HARD_TIMEOUT, true,
                            LOCK_HARD_TIMEOUT, now_ms);
        }
        else if (TimeReached(now_ms, timed_task.dose_deadline_ms))
        {
            FinishTimedTask(TERMINAL_TIMED_DOSE_COMPLETE, false,
                            LOCK_NONE, now_ms);
        }
    }

    Stepper_Task(&z_motor, now_ms);
    ReportAxisCompletion();

    if (binary_session_seen && TimeReached(now_ms, next_periodic_report_ms))
    {
        next_periodic_report_ms = now_ms + APP_PERIODIC_REPORT_MS;
        SendSafetyState(now_ms);
        SendTelemetry(now_ms);
    }
}

void App_UartRxCpltCallback(UART_HandleTypeDef *huart)
{
    uint16_t next_head;

    if (huart != app_uart)
    {
        return;
    }

    uart_rx_armed = false;
    next_head = (uint16_t)((rx_head + 1U) % APP_UART_RX_BUFFER_SIZE);
    if (next_head == rx_tail)
    {
        rx_overflow = true;
    }
    else
    {
        rx_buffer[rx_head] = uart_rx_byte;
        rx_head = next_head;
    }
    ArmUartReceive();
}

void App_UartErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart == app_uart)
    {
        uart_rx_armed = false;
        /* A UART fault during actuation is always fail-closed in this build. */
#if APP_ENABLE_UART_FAULT_FAILSAFE
        if (pump_on || timed_task.active)
        {
            LatchEmergency(LOCK_WATCHDOG,
                           TERMINAL_WATCHDOG_TIMEOUT, HAL_GetTick());
        }
#endif
        ArmUartReceive();
    }
}
