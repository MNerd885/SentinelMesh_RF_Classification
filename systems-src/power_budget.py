def calculate_power_budget():
    # Operating Voltages
    V_max = 3.3 # Volts
    V_checkpoint = 2.2 # Volts
    # Current Draws
    I_mcu_active = 0.005 # 5 mA
    I_mcu_sleep = 0.000003 # 3 uA

    I_radar_active = 0.000300 # ~0.3 mW at 3.3V

    I_lora_tx = 0.100    # 100 mA (SX1262 peak TX)

    # Timing Assumptions
    t_tx_burst = 0.040   # 40 ms per LoRa payload transmission
    radar_duty = 0.10    # Radar active 10% of the time (e.g. 5Hz sweeps)
    mcu_active_time = 0.12 # MCU active 12% of time (sampling + inference)

    # Average Power Calculation (Watts)
    P_mcu = (I_mcu_active * mcu_active_time + I_mcu_sleep * (1 - mcu_active_time)) * V_max
    P_radar = I_radar_active * radar_duty

    # Assuming 1 LoRa TX transmission every 60 seconds
    E_lora_single = V_max * I_lora_tx * t_tx_burst
    P_lora_avg = E_lora_single / 60.0

    P_total_avg = P_mcu + P_radar + P_lora_avg

    # Required Supercapacitor calculation for 5 minutes (300s) blackout buffer
    t_blackout = 300 # seconds
    E_needed = (P_total_avg * t_blackout) + E_lora_single

    # C = 2 * E / (Vmax^2 - Vmin^2)
    C_min = (2 * E_needed) / (V_max**2 - V_checkpoint**2)

    print(f"Average System Power: {P_total_avg * 1000:.3f} mW")
    print(f"Energy for 1 LoRa TX Burst: {E_lora_single * 1000:.2f} mJ")
    print(f"Minimum Supercapacitor Capacitance: {C_min:.4f} Farads")

calculate_power_budget()
