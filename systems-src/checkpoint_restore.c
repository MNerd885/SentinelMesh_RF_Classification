#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define CHECKPOINT_MAGIC 0xDEADBEEF

// Define the fixed memory address in Flash/FRAM reserved for checkpoints
// (Ensure this address range is excluded from program code in your linker script)
#define NVM_CHECKPOINT_ADDR ((uint32_t)0x0803F800) // Example: Last 2KB page of STM32 Flash

typedef enum {
    STATE_IDLE = 0,
    STATE_RADAR_SAMPLING,
    STATE_INFERENCE_RUNNING,
    STATE_LORA_TRANSMITTING
} ExecState_e;

typedef struct {
    uint32_t magic;           // Header flag (0xDEADBEEF)
    ExecState_e current_step; // Execution step
    uint32_t lora_seq_num;    // Packet sequence counter
    float radar_buffer[32];   // Last active radar feature map
} SystemState_t;

// Live runtime state residing in SRAM (marked volatile for ISR safety)
static volatile SystemState_t live_state = {
    .magic = 0,
    .current_step = STATE_IDLE,
    .lora_seq_num = 0,
    .radar_buffer = {0}
};

static volatile bool power_failure_imminent = false;

// Driver-level persistent write wrapper
static void save_state_to_nvm(const SystemState_t *state) {
    // APPROACH A: If using FRAM (e.g., MSP430 FRAM or SPI FRAM)
    // Direct memcpy works because FRAM behaves like non-volatile RAM:
    // memcpy((void*)NVM_CHECKPOINT_ADDR, (void*)state, sizeof(SystemState_t));

    // APPROACH B: If using On-Chip Internal Flash (e.g., STM32)
    // Flash requires unlock -> page erase -> word-by-word write
    /*
    HAL_FLASH_Unlock();
    FLASH_EraseInitTypeDef erase_init = {
        .TypeErase = FLASH_TYPEERASE_PAGES,
        .PageAddress = NVM_CHECKPOINT_ADDR,
        .NbPages = 1
    };
    uint32_t page_error;
    HAL_FLASHEx_Erase(&erase_init, &page_error);

    uint32_t *src = (uint32_t*)state;
    uint32_t *dst = (uint32_t*)NVM_CHECKPOINT_ADDR;
    for (size_t i = 0; i < sizeof(SystemState_t) / sizeof(uint32_t); i++) {
        HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, (uint32_t)(dst + i), src[i]);
    }
    HAL_FLASH_Lock();
    */
}

// Hardware Voltage Interrupt Handler (Triggers when V_DD < V_checkpoint)
void Voltage_Comparator_IRQHandler(void) {
    if (!power_failure_imminent) {
        power_failure_imminent = true;

        // 1. Immediately shut off high-draw peripherals
        // LORA_RADIO_SLEEP();
        // RADAR_DISABLE();

        // 2. Prepare live state payload
        live_state.magic = CHECKPOINT_MAGIC;

        // 3. Write payload directly to persistent Flash/FRAM address
        save_state_to_nvm((const SystemState_t*)&live_state);

        // 4. Enter deep retention mode until voltage starves completely
        // MCU_ENTER_DEEP_SLEEP();
    }
}

// System Boot Sequence (Called early in main)
bool System_Boot_Init(void) {
    // Read directly from non-volatile memory address
    const SystemState_t *nv_ptr = (const SystemState_t*)NVM_CHECKPOINT_ADDR;

    if (nv_ptr->magic == CHECKPOINT_MAGIC) {
        // Recover state from Flash/FRAM into live SRAM variable
        memcpy((void*)&live_state, (const void*)nv_ptr, sizeof(SystemState_t));
        return true; // Warm Boot / Successfully restored
    }

    // Cold boot fallback
    live_state.magic = 0;
    live_state.current_step = STATE_IDLE;
    live_state.lora_seq_num = 0;
    return false; // Cold Boot
}

int main(void) {
    // Hardware initialization routines...

    bool warm_boot = System_Boot_Init();

    if (warm_boot) {
        // Resume runtime loop based on live_state.current_step
    } else {
        // Execute cold startup configuration
    }

    while (1) {
        if (power_failure_imminent) {
            // Halt main thread processing while ISR completes save operation
            continue;
        }

        // Example sensor/inference pipeline updating live state:
        live_state.current_step = STATE_RADAR_SAMPLING;
        // process_radar_frame((float*)live_state.radar_buffer);

        live_state.current_step = STATE_INFERENCE_RUNNING;
        // run_ml_inference();
    }

    return 0;
}
