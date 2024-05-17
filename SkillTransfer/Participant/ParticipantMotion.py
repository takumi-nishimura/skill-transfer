import csv
import threading
import time

import numpy as np

from BendingSensor.BendingSensorManager import BendingSensorManager
from MotionFilter.MotionFilter import MotionFilter
from OptiTrack.OptiTrackStreaming import OptiTrackStreamingManager

# ----- Custom class ----- #
from UDP.UDPManager import UDPManager

# ----- Numeric range remapping ----- #
targetMin = 150
targetMax = 850
originalMin = 0
originalMax = 1

# ----- For many bendingsensors (Arduinoの値を見て変更) ----- #
# bendingSensorMin = [1600,2000]
# bendingSensorMax = [2400,2500]

# ----- Settings: Recorded motion data ----- #
recordedMotionPath = "RecordedMotion/"
recordedMotionFileName = "Transform_Participant_"
recordedGripperValueFileName = "GripperValue_"


class ParticipantMotionManager:
    def __init__(
        self,
        defaultParticipantNum: int,
        motionInputSystem: str = "optitrack",
        mocapServer: str = "",
        mocapLocal: str = "",
        gripperInputSystem: str = "bendingsensor",
        bendingSensorNum: int = 1,
        BendingSensor_ConnectionMethod: str = "wireless",
        recordedGripperValueNum: int = 0,
        bendingSensorUdpIpAddress: str = "192.168.80.142",
        bendingSensorUdpPort: list = [9000, 9001],
        bendingSensorSerialCOMs: list = [],
    ) -> None:

        self.defaultParticipantNum = defaultParticipantNum
        self.motionInputSystem = motionInputSystem
        self.gripperInputSystem = gripperInputSystem
        self.bendingSensorNum = bendingSensorNum
        self.recordedGripperValueNum = recordedGripperValueNum
        self.udpManager = None
        self.recordedMotion = {}
        self.recordedGripperValue = {}
        self.recordedMotionLength = []
        self.InitBendingSensorValues = []

        n = 2
        fp = 10
        fs = 700
        self.filter_FB = MotionFilter()
        self.filter_FB.InitLowPassFilterWithOrder(fs, fp, n)

        self.get_gripperValue_1_box = [[0]] * n
        self.get_gripperValue_1_filt_box = [[0]] * n
        self.get_gripperValue_2_box = [[0]] * n
        self.get_gripperValue_2_filt_box = [[0]] * n

        # ----- Initialize participants' motion input system ----- #
        if motionInputSystem == "optitrack":
            self.optiTrackStreamingManager = OptiTrackStreamingManager(
                defaultParticipantNum=defaultParticipantNum,
                mocapServer=mocapServer,
                mocapLocal=mocapLocal,
            )

            # ----- Start streaming from OptiTrack ----- #
            streamingThread = threading.Thread(
                target=self.optiTrackStreamingManager.stream_run
            )
            streamingThread.setDaemon(True)
            streamingThread.start()

        # ----- Initialize gripper control system ----- #
        if gripperInputSystem == "bendingsensor":
            self.bendingSensors = []

            if BendingSensor_ConnectionMethod == "wireless":
                print("wireless")
                self.ip = [bendingSensorUdpIpAddress, bendingSensorUdpIpAddress]
                self.port = bendingSensorUdpPort
            elif BendingSensor_ConnectionMethod == "wired":
                print("wired")
                self.ip = bendingSensorSerialCOMs
                self.port = bendingSensorUdpPort

            for i in range(bendingSensorNum):
                bendingSensorManager = BendingSensorManager(
                    BendingSensor_connectionmethod=BendingSensor_ConnectionMethod,
                    ip=self.ip[i],
                    port=self.port[i],
                )
                self.bendingSensors.append(bendingSensorManager)

                # ----- Start receiving bending sensor value from UDP socket ----- #
                bendingSensorThread = threading.Thread(
                    target=bendingSensorManager.StartReceiving
                )
                bendingSensorThread.setDaemon(True)
                bendingSensorThread.start()

            # ----- Set init value ----- #
            self.SetInitialBendingValue()

    def SetInitialBendingValue(self):
        """
        Set init bending value
        """

        if self.gripperInputSystem == "bendingsensor":
            self.InitBendingSensorValues = []

            for i in range(self.bendingSensorNum):
                self.InitBendingSensorValues.append(self.bendingSensors[i].bendingValue)

    def LocalPosition(self, loopCount: int = 0):
        """
        Local position

        Parameters
        ----------
        loopCount: (Optional) int
            For recorded motion.
            Count of loop.

        Returns
        ----------
        participants' local position: dict
        {'participant1': [x, y, z]}
        unit: [m]
        """

        dictPos = {}
        if self.motionInputSystem == "optitrack":
            dictPos = self.optiTrackStreamingManager.position

        return dictPos

    def LocalRotation(self, loopCount: int = 0):
        """
        Local rotation

        Parameters
        ----------
        loopCount: (Optional) int
            For recorded motion.
            Count of loop.

        Returns
        ----------
        participants' local rotation: dict
        {'participant1': [x, y, z, w] or [x, y, z]}
        """

        dictRot = {}
        if self.motionInputSystem == "optitrack":
            dictRot = self.optiTrackStreamingManager.rotation

        return dictRot

    def GripperControlValue(self, weight: list, loopCount: int = 0):
        """
        Value for control of the xArm gripper

        Parameters
        ----------
        loopCount: (Optional) int
            For recorded motion.
            Count of loop.

        Returns
        ----------
        Value for control of the xArm gripper: dict
        {'gripperValue1': float value}
        """

        if self.gripperInputSystem == "bendingsensor":
            dictGripperValue = {}
            dictbendingVal = {}
            for i in range(self.bendingSensorNum):
                dictbendingVal["gripperValue" + str(i + 1)] = self.bendingSensors[
                    i
                ].bendingValue

            if self.bendingSensorNum == 2:
                bendingValueNorm1 = dictbendingVal["gripperValue1"]
                print(bendingValueNorm1)
                bendingValueNorm2 = dictbendingVal["gripperValue2"]
            elif self.bendingSensorNum == 4:
                bendingValueNorm1 = (
                    dictbendingVal["gripperValue1"] * weight[0]
                    + dictbendingVal["gripperValue3"] * weight[2]
                )
                bendingValueNorm2 = (
                    dictbendingVal["gripperValue2"] * weight[1]
                    + dictbendingVal["gripperValue4"] * weight[3]
                )
            elif self.bendingSensorNum == 6:
                bendingValueNorm1 = (
                    dictbendingVal["gripperValue1"] * weight[0]
                    + dictbendingVal["gripperValue3"] * weight[2]
                    + dictbendingVal["gripperValue5"] * weight[4]
                )
                bendingValueNorm2 = (
                    dictbendingVal["gripperValue2"] * weight[1]
                    + dictbendingVal["gripperValue4"] * weight[3]
                    + dictbendingVal["gripperValue6"] * weight[5]
                )

            GripperValue1 = bendingValueNorm1 * (targetMax - targetMin) + targetMin
            GripperValue2 = bendingValueNorm2 * (targetMax - targetMin) + targetMin

            if GripperValue1 > targetMax:
                GripperValue1 = targetMax
            if GripperValue2 > targetMax:
                GripperValue2 = targetMax
            if GripperValue1 < targetMin:
                GripperValue1 = targetMin
            if GripperValue2 < targetMin:
                GripperValue2 = targetMin

            # ----- lowpass filter for gripper1 ----- #
            self.get_gripperValue_1_box.append([GripperValue1])
            get_gripperValue_1_filt = self.filter_FB.lowpass2(
                self.get_gripperValue_1_box, self.get_gripperValue_1_filt_box
            )
            self.get_gripperValue_1_filt_box.append(get_gripperValue_1_filt)
            del self.get_gripperValue_1_box[0]
            del self.get_gripperValue_1_filt_box[0]

            # ----- lowpass filter for gripper2 ----- #
            self.get_gripperValue_2_box.append([GripperValue2])
            get_gripperValue_2_filt = self.filter_FB.lowpass2(
                self.get_gripperValue_2_box, self.get_gripperValue_2_filt_box
            )
            self.get_gripperValue_2_filt_box.append(get_gripperValue_2_filt)
            del self.get_gripperValue_2_box[0]
            del self.get_gripperValue_2_filt_box[0]

            dictGripperValue["gripperValue1"] = get_gripperValue_1_filt
            dictGripperValue["gripperValue2"] = get_gripperValue_2_filt

        return dictGripperValue, dictbendingVal
