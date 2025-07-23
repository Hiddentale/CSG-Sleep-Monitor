import machine
import time

led = machine.Pin(25, machine.Pin.OUT)

while True:
    led.toggle()
    print("LED toggled")
    time.sleep(1)