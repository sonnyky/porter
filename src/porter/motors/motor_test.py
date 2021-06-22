import RPi.GPIO as GPIO
import time

servoPIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 100) # GPIO 17 for PWM with 50Hz
p.start(0) # Initialization
try:
  servo_current_angle = 0
  servo_desired_angle = 30
  x = True
  duty_cycle = 5.5
  go_up = True
  go_down = False
  while (x == True) :
    
    p.ChangeDutyCycle(11.5)
    time.sleep(0.5)
    print(duty_cycle)
    

  print(duty_cycle)
  p.stop()
  GPIO.cleanup()
except KeyboardInterrupt:
  p.stop()
  GPIO.cleanup()
