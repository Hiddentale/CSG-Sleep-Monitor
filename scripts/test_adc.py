import machine
import time

analog_to_digital_converter = machine.ADC(26)

while True:
    reading = analog_to_digital_converter.read_u16()
    voltage = reading * 3.3 / 65535
    print(f"ADC: {reading}, Voltage: {voltage:.3f}V")
    time.sleep(0.5)
    