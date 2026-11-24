int main(void) {
    // 1. Check for valid warm boot checkpoint from non-volatile storage
    bool is_warm_boot = System_Boot_Init();

    // 2. Initialize Hardware Peripherals
    radar_init();

    if (!is_warm_boot) {
        // Cold start sequence
        live_state.current_step = STATE_IDLE;
        live_state.lora_seq_num = 0;
    }

    while (1) {
        // If energy brownout is triggered by ISR, pause execution
        if (power_failure_imminent) {
            continue;
        }

        switch (live_state.current_step) {
            case STATE_IDLE:
                live_state.current_step = STATE_RADAR_SAMPLING;
                break;

            case STATE_RADAR_SAMPLING:
                // Fetch radar sweep data
                radar_read_frame((float*)live_state.radar_buffer, 32);
                live_state.current_step = STATE_INFERENCE_RUNNING;
                break;

            case STATE_INFERENCE_RUNNING:
                // ML model's processes live_state.radar_buffer locally
                // If a fall or state change is detected:
                //     live_state.current_step = STATE_LORA_TRANSMITTING;
                // Else:
                //     live_state.current_step = STATE_IDLE;
                break;

            case STATE_LORA_TRANSMITTING:
                {
                    uint8_t payload[2] = { 0x01, (uint8_t)live_state.lora_seq_num };
                    lora_send_payload(payload, sizeof(payload));
                    live_state.lora_seq_num++;
                    live_state.current_step = STATE_IDLE;
                }
                break;

            default:
                live_state.current_step = STATE_IDLE;
                break;
        }
    }
}
