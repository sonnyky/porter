import RPi.GPIO as GPIO
import time
import math
import numpy as np


class MotorController:
    def __init__(self):
        self.servoPIN_A = 27
        self.servoPIN_B = 18
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servoPIN_A, GPIO.OUT)
        GPIO.setup(self.servoPIN_B, GPIO.OUT)

        # OAK-D specifications here : https://www.robotshop.com/en/opencv-oak-d-3d-ai-camera-kit.html
        self.HFOV = 35.6 # degrees, half of the HFOV in the spec sheet
        self.turnThresholdPercentageRatio = 0.3

        self.pwm_A = GPIO.PWM(self.servoPIN_A, 100)  # GPIO 17 for PWM with 50Hz
        self.pwm_B = GPIO.PWM(self.servoPIN_B, 100)  # GPIO 17 for PWM with 50Hz
        self.leftDutyCycle = 0
        self.rightDutyCycle = 0
        self.leftTime = 0.1
        self.rightTime = 0.1
        self.constantSpeed = 11.5
        self.thresholdDistance = 1500  # in mm, due to angle this value is bigger for now
        self.pwm_A.start(0)
        self.pwm_B.start(0)

    def testPlugin(self):
        print("Motor controller connected")

    def runLeftMotors(self):
        self.pwm_A.ChangeDutyCycle(self.leftDutyCycle)
        time.sleep(self.leftTime)

    def runRightMotors(self):
        self.pwm_B.ChangeDutyCycle(self.rightDutyCycle)
        time.sleep(self.rightTime)

    def stopMotors(self):
        self.leftDutyCycle = 0
        self.rightDutyCycle = 0
        self.pwm_A.ChangeDutyCycle(0)
        self.pwm_B.ChangeDutyCycle(0)

    def directionControl(self, pos_x, pos_y, pos_z):

        # check what kind of values are returned if we lost track
        print("Pos: " + str(pos_x) + ", " + str(pos_y) + ", " + str(pos_z))

        turnThreshold = self.calculateTurnThreshold(pos_z)
        print("Turn threshold: " + str(turnThreshold))

        if pos_z < self.thresholdDistance:
            self.stopMotors()
        elif pos_x > turnThreshold:
            self.rightDutyCycle = 0
            self.leftDutyCycle = 11.5
            self.leftTime = 0.5
           
            print("turning right")
        elif pos_x < -1 * turnThreshold:
            self.leftDutyCycle = 0
            self.rightDutyCycle = 11.5
            self.rightTime = 0.5
           
            print("turning left")
        else:
            self.leftDutyCycle = self.constantSpeed
            self.rightDutyCycle = self.constantSpeed
            
            print("going straight")
            
        self.runLeftMotors()
        self.runRightMotors()
        self.leftTime = 0.1
        self.rightTime = 0.1

        print(str(self.rightDutyCycle) + ", " + str(self.leftDutyCycle))

    def calculateTurnThreshold(self, pos_z):
        tangent = math.tan(self.HFOV * math.pi / 180)
        print("tangent: " + str(tangent))
        maxViewableSide = tangent  * pos_z
        print("Max side: " + str(maxViewableSide))
        return self.turnThresholdPercentageRatio * maxViewableSide

    def completeStop(self):
        self.stopMotors()
        self.pwm_B.stop()
        self.pwm_A.stop()
        GPIO.cleanup()


mc = MotorController()
try:
    while True:
        mc.runLeftMotors()
        mc.runRightMotors()

except KeyboardInterrupt:
    mc.completeStop()
