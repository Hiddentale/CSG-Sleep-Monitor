import machine
import time

analog_to_digital_converter = machine.ADC(26)
lead_off_plus = machine.Pin(14, machine.Pin.IN)
lead_off_min = machine.Pin(15, machine.Pin.IN)

def read_ecg_data():
    if lead_off_plus.value() == 1 or lead_off_min.value() == 1:
        return None
    
    raw_ecg_data = analog_to_digital_converter.read_u16()
    maximum_voltage = 3.3
    number_of_possible_ecg_states = 65535
    voltage = raw_ecg_data * maximum_voltage / number_of_possible_ecg_states
    return voltage

while True:
    ecg_data = read_ecg_data()
    if ecg_data is not None:
        print(f"ECG: {ecg_data:.3f}V")
    else:
        print("Electrodes malfunctioning or not connected")
    time.sleep(0.1)