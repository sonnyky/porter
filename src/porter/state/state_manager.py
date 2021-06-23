from .tracking_state import TrackingState
from .reid_state import ReidState
from .registration_state import RegistrationState
from porter.motors.motor import MotorController


class StateManager:
    def __init__(self, device):
        self.device = device

        # the motor controller has to be instantiated before the states
        self.mc = MotorController()

        self.tracking_state = TrackingState(device, self)
        self.reid_state = ReidState(device, self)
        self.registration_state = RegistrationState(device, self)
        self.current_state = self.registration_state

        self.targetPerson = {}
        self.targetPersonId = 0
        self.nextId = 0

    def GetTargetPersonId(self):
        return self.targetPersonId

    def SetTargetPersonId(self, newId):
        self.targetPersonId = newId

    def GetMotorController(self):
        return self.mc

    def SetTargetPerson(self, targetPerson):
        self.targetPerson = targetPerson
        self.changeCurrentState(self.tracking_state)

    def GetTargetPerson(self):
        return self.targetPerson

    def TargetLost(self):
        self.changeCurrentState(self.reid_state)

    def TargetReidentified(self):
        self.changeCurrentState(self.tracking_state)

    def getCurrentState(self):
        return self.current_state

    def changeCurrentState(self, newState):
        self.current_state = newState

    def update(self, trackletsData, frame):
        self.current_state.updateData(trackletsData, frame)
        self.current_state.update()
