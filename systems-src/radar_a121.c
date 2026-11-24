#include <stdint.h>
#include <stdbool.h>

// Low-Level Driver Pin Handles
#define RADAR_CS_LOW()   HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET)
#define RADAR_CS_HIGH()  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET)
#define RADAR_ENABLE()   HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, GPIO_PIN_SET)
#define RADAR_DISABLE()  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, GPIO_PIN_RESET)

void radar_init(void) {
    RADAR_ENABLE();
    // Allow radar internal LDO to stabilize (~2ms)
    // Delay_ms(2);
}

bool radar_read_frame(float *feature_buffer, uint16_t buffer_len) {
    RADAR_CS_LOW();

    // SPI read sequence: Command opcode + register address
    uint8_t tx_cmd[2] = { 0x02, 0x00 }; // Read burst command
    // SPI_TransmitReceive_DMA(SPI_BUS_1, tx_cmd, feature_buffer, buffer_len);

    RADAR_CS_HIGH();
    return true;
}
