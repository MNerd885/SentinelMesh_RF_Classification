#define LORA_CS_LOW()    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_RESET)
#define LORA_CS_HIGH()   HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_SET)

// Wait for SX1262 internal state machine to clear BUSY flag
static void lora_wait_busy(void) {
    while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_4) == GPIO_PIN_SET) {
        // Spin lock until BUSY pin goes LOW
    }
}

void lora_send_payload(const uint8_t *data, uint8_t length) {
    lora_wait_busy();

    LORA_CS_LOW();

    // Opcode 0x0E: WriteBuffer in SX1262 command set
    uint8_t cmd[2] = { 0x0E, 0x00 }; // Opcode + Buffer offset
    // SPI_Transmit(SPI_BUS_2, cmd, 2);
    // SPI_Transmit(SPI_BUS_2, (uint8_t*)data, length);

    LORA_CS_HIGH();

    // Trigger Tx Opcode (0x83) with timeout parameters
    lora_wait_busy();
    LORA_CS_LOW();
    uint8_t tx_cmd[4] = { 0x83, 0x00, 0x00, 0x00 };
    // SPI_Transmit(SPI_BUS_2, tx_cmd, 4);
    LORA_CS_HIGH();
}
