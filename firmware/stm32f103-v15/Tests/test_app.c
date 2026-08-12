#include "app.h"
#include "app_config.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

GPIO_TypeDef mock_gpio_a = {1U, 0U};
GPIO_TypeDef mock_gpio_b = {2U, 0U};

static uint32_t mock_tick;
static uint16_t gpio_a_state;
static uint16_t gpio_b_state;
static uint8_t *uart_rx_target;
static char uart_tx_log[4096];
static size_t uart_tx_length;
static uint8_t z_pattern_log[32];
static size_t z_pattern_length;

void HAL_GPIO_WritePin(GPIO_TypeDef *port, uint16_t pin, GPIO_PinState state)
{
    uint16_t *value = (port == GPIOA) ? &gpio_a_state : &gpio_b_state;
    if (state == GPIO_PIN_SET)
    {
        *value |= pin;
    }
    else
    {
        *value &= (uint16_t)~pin;
    }
    port->ODR = *value;

    /*
     * Stepper_WritePattern writes IND last, so sampling after the Z IND write
     * captures the complete four-bit pattern presented to the driver.
     */
    if ((port == GPIOA) && (pin == Z_IND_Pin) &&
        (z_pattern_length < sizeof(z_pattern_log)))
    {
        z_pattern_log[z_pattern_length++] =
            (uint8_t)(gpio_a_state & 0x0FU);
    }
}

HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *huart, uint8_t *data,
                                    uint16_t length, uint32_t timeout)
{
    (void)huart;
    (void)timeout;
    assert((uart_tx_length + length) < sizeof(uart_tx_log));
    memcpy(&uart_tx_log[uart_tx_length], data, length);
    uart_tx_length += length;
    uart_tx_log[uart_tx_length] = '\0';
    return HAL_OK;
}

HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *huart, uint8_t *data,
                                      uint16_t length)
{
    (void)huart;
    assert(length == 1U);
    uart_rx_target = data;
    return HAL_OK;
}

uint32_t HAL_GetTick(void)
{
    return mock_tick;
}

static void FeedCommand(UART_HandleTypeDef *uart, const char *text)
{
    while (*text != '\0')
    {
        *uart_rx_target = (uint8_t)*text++;
        App_UartRxCpltCallback(uart);
    }
    App_Task();
}

static void RunMilliseconds(uint32_t duration_ms)
{
    uint32_t i;
    for (i = 0U; i < duration_ms; ++i)
    {
        ++mock_tick;
        App_Task();
    }
}

static int LogContains(const char *text)
{
    size_t text_length = strlen(text);
    size_t i;

    if (text_length == 0U || text_length > uart_tx_length)
    {
        return 0;
    }
    for (i = 0U; i + text_length <= uart_tx_length; ++i)
    {
        if (memcmp(&uart_tx_log[i], text, text_length) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static void FeedBytes(UART_HandleTypeDef *uart, const uint8_t *data,
                      size_t length)
{
    size_t i;
    for (i = 0U; i < length; ++i)
    {
        *uart_rx_target = data[i];
        App_UartRxCpltCallback(uart);
    }
    App_Task();
}

static size_t BuildFrame(uint8_t type, const uint8_t *payload,
                         uint8_t payload_length, uint8_t *frame)
{
    size_t i;
    uint8_t sum;

    frame[0] = 0xAAU;
    frame[1] = 0x55U;
    frame[2] = type;
    frame[3] = payload_length;
    sum = (uint8_t)(0xAAU + 0x55U + type + payload_length);
    for (i = 0U; i < payload_length; ++i)
    {
        frame[4U + i] = payload[i];
        sum = (uint8_t)(sum + payload[i]);
    }
    frame[4U + payload_length] = sum;
    return (size_t)payload_length + 5U;
}

static void PutU16(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)(value >> 8U);
}

static void PutU32(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)(value >> 8U);
    data[2] = (uint8_t)(value >> 16U);
    data[3] = (uint8_t)(value >> 24U);
}

static int TxHasFrameType(uint8_t type)
{
    size_t i;
    for (i = 0U; i + 3U < uart_tx_length; ++i)
    {
        if ((uint8_t)uart_tx_log[i] == 0xAAU &&
            (uint8_t)uart_tx_log[i + 1U] == 0x55U &&
            (uint8_t)uart_tx_log[i + 2U] == type)
        {
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    UART_HandleTypeDef uart = {1U};
    uint8_t payload[23] = {0};
    uint8_t frame[64];
    size_t frame_length;

    App_Init(&uart);
    assert(LogContains("READY,STM32_IRRIGATION," APP_FIRMWARE_VERSION
                       ",PB6_ACTIVE_LOW_OD,Z3_DOWN_ONLY,MANUAL_RETURN,"
                       "BOOT_LOCKED,OUTPUTS_OFF\r\n"));
    assert((gpio_b_state & PUMP_Pin) != 0U);
    assert((gpio_b_state & GPIO_PIN_7) == 0U);

    FeedCommand(&uart, "PING\r\n");
    assert(LogContains("ACK,PONG\r\n"));

    FeedCommand(&uart, "VERSION\r\n");
    assert(LogContains("VERSION," APP_FIRMWARE_VERSION
                       ",HW=F103C8,PUMP=PB6,RELAY=ACTIVE_LOW_OD,"
                       "MOTION=Z3_DOWN_ONLY,RETURN=MANUAL"));

    FeedCommand(&uart, "IOSTATUS\r\n");
    assert(LogContains("IOSTATUS,PA=0x00,Z=0x0,"
                       "PB6=1,PLOG=0\r\n"));

    FeedCommand(&uart, "MOVE,X,4,20\r\n");
    assert(LogContains("ERR,MOTION_EXTENSION_LOCKED\r\n"));
    FeedCommand(&uart, "COIL,X,1\r\n");
    assert(LogContains("ERR,MOTION_EXTENSION_LOCKED\r\n"));
    FeedCommand(&uart, "MODE,X,3\r\n");
    assert(LogContains("ERR,MOTION_EXTENSION_LOCKED\r\n"));

    FeedCommand(&uart, "COIL,Z,1\r\n");
    assert(LogContains("ERR,MOTION_EXTENSION_LOCKED\r\n"));
    assert((gpio_a_state & (Z_INA_Pin | Z_INB_Pin |
                            Z_INC_Pin | Z_IND_Pin)) == 0U);
    FeedCommand(&uart, "IOSTATUS\r\n");
    assert(LogContains("IOSTATUS,PA=0x00,Z=0x0,"
                       "PB6=1,PLOG=0\r\n"));

    FeedCommand(&uart, "MOVE,Z,4,5\r\n");
    assert(LogContains("ERR,MOTION_EXTENSION_LOCKED\r\n"));
    assert((gpio_a_state & (Z_INA_Pin | Z_INB_Pin |
                            Z_INC_Pin | Z_IND_Pin)) == 0U);

    /* Sand/non-target is a fail-closed no-motion preset. */
    FeedCommand(&uart, "DEPTH,0\r\n");
    assert(LogContains("ACK,DEPTH,0,HOLD,NO_MOTION\r\n"));
    assert((gpio_a_state & (Z_INA_Pin | Z_INB_Pin |
                            Z_INC_Pin | Z_IND_Pin)) == 0U);
    FeedCommand(&uart, "DEPTH,4\r\n");
    assert(LogContains("ERR,BAD_DEPTH_PRESET\r\n"));
    FeedCommand(&uart, "ZHOME,NO\r\n");
    assert(LogContains("ERR,ZHOME_REQUIRES_CONFIRM\r\n"));

    FeedCommand(&uart, "PUMP,2,ON\r\n");
    assert(LogContains("ERR,BAD_PUMP\r\n"));
    assert((gpio_b_state & PUMP_Pin) != 0U);

    FeedCommand(&uart, "WATER,3,1,5,1\r\n");
    assert(LogContains("ERR,UNKNOWN_COMMAND\r\n"));

    FeedCommand(&uart, "PUMP,1,ON\r\n");
    assert(LogContains("ERR,UNBOUNDED_PUMP_DISABLED\r\n"));
    assert((gpio_b_state & PUMP_Pin) != 0U);
    FeedCommand(&uart, "IOSTATUS\r\n");
    assert(LogContains("PB6=1,PLOG=0\r\n"));

    FeedCommand(&uart, "PUMP,1,OFF\r\n");
    assert(LogContains("ACK,PUMP,1,OFF\r\n"));
    assert((gpio_b_state & PUMP_Pin) != 0U);
    FeedCommand(&uart, "IOSTATUS\r\n");
    assert(LogContains("PB6=1,PLOG=0\r\n"));

    /* Query identity, establish heartbeat, and explicitly clear the boot lock. */
    PutU16(payload, 1U);
    frame_length = BuildFrame(0x12U, payload, 2U, frame);
    FeedBytes(&uart, frame, frame_length);
    assert(TxHasFrameType(0x81U));

    PutU16(payload, 2U);
    payload[2] = 0U;
    frame_length = BuildFrame(0xFFU, payload, 3U, frame);
    FeedBytes(&uart, frame, frame_length);

    PutU16(payload, 3U);
    frame_length = BuildFrame(0x11U, payload, 2U, frame);
    FeedBytes(&uart, frame, frame_length);

    /*
     * A manual home confirmation admits one downward preset.  The heartbeat
     * watchdog must de-energize all four PA0..PA3 coils if the host disappears.
     */
    FeedCommand(&uart, "ZHOME,CONFIRM\r\n");
    assert(LogContains("ACK,ZHOME,CONFIRMED,MANUAL_OBSERVATION\r\n"));
    FeedCommand(&uart, "DEPTH,1\r\n");
    assert(LogContains("ACK,DEPTH,1,DOWN,STEPS=1024,"
                       "MANUAL_RETURN_REQUIRED\r\n"));
    assert((gpio_a_state & (Z_INA_Pin | Z_INB_Pin |
                            Z_INC_Pin | Z_IND_Pin)) != 0U);
    FeedCommand(&uart, "DEPTH,2\r\n");
    assert(LogContains("ERR,AXIS_BUSY\r\n"));
    RunMilliseconds(APP_HEARTBEAT_TIMEOUT_MS + 1U);
    assert((gpio_a_state & (Z_INA_Pin | Z_INB_Pin |
                            Z_INC_Pin | Z_IND_Pin)) == 0U);
    FeedCommand(&uart, "DEPTH,2\r\n");
    assert(LogContains("ERR,SAFETY_LOCKED\r\n"));

    /* Re-establish the binary safety session for the bounded pump task. */
    PutU16(payload, 4U);
    payload[2] = 0U;
    frame_length = BuildFrame(0xFFU, payload, 3U, frame);
    FeedBytes(&uart, frame, frame_length);

    PutU16(payload, 5U);
    frame_length = BuildFrame(0x11U, payload, 2U, frame);
    FeedBytes(&uart, frame, frame_length);

    /* A used descent cannot be repeated until the operator confirms home. */
    FeedCommand(&uart, "DEPTH,2\r\n");
    assert(LogContains("ERR,ZHOME_NOT_CONFIRMED\r\n"));
    FeedCommand(&uart, "ZHOME,CONFIRM\r\n");
    FeedCommand(&uart, "STATUS\r\n");
    assert(LogContains("ZHOME=1,ZUSED=0,ZLEVEL=0,ZSTEPS=0\r\n"));

    /* The sole pump-on path is a bounded binary timed task. */
    memset(payload, 0, sizeof(payload));
    PutU32(payload, 1U);
    PutU16(payload + 4U, 6U);
    payload[6] = 1U;
    PutU32(payload + 7U, 100U);
    PutU32(payload + 11U, 500U);
    frame_length = BuildFrame(0x22U, payload, 23U, frame);
    FeedBytes(&uart, frame, frame_length);
    assert((gpio_b_state & PUMP_Pin) == 0U);
    RunMilliseconds(99U);
    assert((gpio_b_state & PUMP_Pin) == 0U);
    RunMilliseconds(1U);
    assert((gpio_b_state & PUMP_Pin) != 0U);
    assert(TxHasFrameType(0x84U));

    /* Replaying the same global sequence must not re-energize PB6. */
    FeedBytes(&uart, frame, frame_length);
    assert((gpio_b_state & PUMP_Pin) != 0U);

    /* Heartbeat expiry keeps PB6 off and re-latches the controller. */
    RunMilliseconds(APP_HEARTBEAT_TIMEOUT_MS + 1U);
    assert((gpio_b_state & PUMP_Pin) != 0U);

    FeedCommand(&uart, "STOP\r\n");
    assert(LogContains("ACK,STOP,LOCKED\r\n"));
    assert((gpio_b_state & PUMP_Pin) != 0U);

    puts("All host-side firmware tests passed.");
    return 0;
}
