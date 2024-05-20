# -----------------------------------------------------------------------
# Author:   Takayoshi Hagiwara (KMD)
# Created:  2021/8/19
# Summary:  Robot arm motion control manager
# -----------------------------------------------------------------------

import datetime
import pprint
import threading
import time
from ctypes import windll
from datetime import datetime
from enum import Flag
from turtle import right

import numpy as np
from cv2 import transform
from FileIO.FileIO import FileIO
# ----- Custom class ----- #
from Gripper.Gripper import BendingSensorManager
from matplotlib.pyplot import flag
from Participant.ParticipantMotion import ParticipantMotion
from Recorder.DataRecord import DataRecordManager
from Robot.CAMotion import CAMotion
from Robot.xArmTransform import xArmTransform
from xarm.wrapper import XArmAPI

# ---------- Settings: Input mode ---------- #
motionDataInputMode = "optitrack"
gripperDataInputMode = "bendingsensor"

class ProcessorClass:
    def __init__(self) -> None:
        fileIO = FileIO()
        dat = fileIO.Read("settings.csv", ",")

        xArmIP_left = [addr for addr in dat if "xArmIPAddress_left" in addr[0]][0][1]
        initialpos_left = [addr for addr in dat if "initialpos_left," in addr[0]]
        initialrot_left = [addr for addr in dat if "initialrot_left," in addr[0]]

        xArmIP_right = [addr for addr in dat if "xArmIPAddress_right" in addr[0]][0][1]
        initialpos_right = [addr for addr in dat if "initialpos_right," in addr[0]]
        initialrot_right = [addr for addr in dat if "initialrot_right," in addr[0]]

        wirelessIP = [addr for addr in dat if "wirelessIPAddress" in addr[0]][0][1]
        localIP = [addr for addr in dat if "localIPAddress" in addr[0]][0][1]
        motiveserverIP = [addr for addr in dat if "motiveServerIPAddress" in addr[0]][0][1]
        motivelocalIP = [addr for addr in dat if "motiveLocalIPAddress" in addr[0]][0][1]
        frameRate = [addr for addr in dat if "frameRate" in addr[0]][0][1]

        bendingSensorPortParticipant1 = [addr for addr in dat if "bendingSensorPortParticipant1" in addr[0]][0][1]
        bendingSensorPortParticipant2 = [addr for addr in dat if "bendingSensorPortParticipant2" in addr[0]][0][1]
        bendingSensorPortParticipant3 = [addr for addr in dat if "bendingSensorPortParticipant3" in addr[0]][0][1]
        bendingSensorPortParticipant4 = [addr for addr in dat if "bendingSensorPortParticipant4" in addr[0]][0][1]
        bendingSensorPortParticipant5 = [addr for addr in dat if "bendingSensorPortParticipant5" in addr[0]][0][1]
        bendingSensorPortParticipant6 = [addr for addr in dat if "bendingSensorPortParticipant6" in addr[0]][0][1]

        bendingSensorCom1 = [addr for addr in dat if "bendingSensorCom1" in addr[0]][0][1]
        bendingSensorCom2 = [addr for addr in dat if "bendingSensorCom2" in addr[0]][0][1]
        bendingSensorCom3 = [addr for addr in dat if "bendingSensorCom3" in addr[0]][0][1]
        bendingSensorCom4 = [addr for addr in dat if "bendingSensorCom4" in addr[0]][0][1]
        bendingSensorCom5 = [addr for addr in dat if "bendingSensorCom5" in addr[0]][0][1]
        bendingSensorCom6 = [addr for addr in dat if "bendingSensorCom6" in addr[0]][0][1]

        isExportData = [addr for addr in dat if "isExportData" in addr[0]][0][1]
        dirPath = [addr for addr in dat if "dirPath" in addr[0]][0][1]

        participantNum = [addr for addr in dat if "participantNum" in addr[0]][0][1]
        gripperNum = [addr for addr in dat if "gripperNum" in addr[0]][0][1]
        otherRigidBodyNum = [addr for addr in dat if "otherRigidBodyNum" in addr[0]][0][1]
        robotNum = [addr for addr in dat if "robotNum" in addr[0]][0][1]

        weightListPos = [addr for addr in dat if "weightListPos" in addr[0]]
        weightListRot = [addr for addr in dat if "weightListRot" in addr[0]]
        weightGripperList = [addr for addr in dat if "weightGripperList" in addr[0]]

        self.xArmIpAddress_left = xArmIP_left
        self.initialpos_left = initialpos_left
        self.initislrot_left = initialrot_left

        self.xArmIpAddress_right = xArmIP_right
        self.initialpos_right = initialpos_right
        self.initislrot_right = initialrot_right

        self.wirelessIpAddress = wirelessIP
        self.localIpAddress = localIP
        self.motiveserverIpAddress = motiveserverIP
        self.motivelocalIpAddress = motivelocalIP
        self.frameRate = frameRate

        self.bendingSensorPorts = [int(bendingSensorPortParticipant1), int(bendingSensorPortParticipant2), int(bendingSensorPortParticipant3), int(bendingSensorPortParticipant4), int(bendingSensorPortParticipant5), int(bendingSensorPortParticipant6)]
        self.bendingSensorComs = [bendingSensorCom1, bendingSensorCom2, bendingSensorCom3, bendingSensorCom4, bendingSensorCom5, bendingSensorCom6]

        self.isExportData = isExportData
        self.dirPath = dirPath

        self.participantNum = participantNum
        self.gripperNum = gripperNum
        self.otherRigidBodyNum = otherRigidBodyNum
        self.robotNum = robotNum

        self.weightListPos = weightListPos
        self.weightListRot = weightListRot
        self.weightGripperList = weightGripperList

        self.participantname = "卒論"
        # self.condition = input('---実験条件---\nFB無し-->A, 相手-->B, ロボット-->C   :')
        # self.number = input('---試行回数---\n何回目   :')
        self.condition = "1"
        self.number = "1"

    def mainloop(self, isFixedFrameRate: bool = False, isChangeOSTimer: bool = False, isExportData: bool = False, isEnablexArm: bool = True):
        """
        Send the position and rotation to the xArm
        """

        # ----- Change OS timer ----- #
        if isFixedFrameRate and isChangeOSTimer:
            windll.winmm.timeBeginPeriod(1)

        # ----- Process info ----- #
        self.loopCount = 0
        self.taskTime = []
        self.errorCount = 0
        taskStartTime = 0

        # ----- Set loop time from frameRate ----- #
        loopTime = 1 / self.frameRate
        loopStartTime = 0
        processDuration = 0
        listFrameRate = []
        if isFixedFrameRate:
            print("Use fixed frame rate > " + str(self.frameRate) + "[fps]")

        # ----- Instantiating custom classes ----- #
        caMotion = CAMotion(defaultParticipantNum=self.participantNum, otherRigidBodyNum=self.otherRigidBodyNum)
        transform_left = xArmTransform(initpos=self.initialpos_left, initrot=self.initislrot_left, mount="left")
        transform_right = xArmTransform(initpos=self.initialpos_right, initrot=self.initislrot_right, mount="right")
        dataRecordManager = DataRecordManager(participantNum=self.participantNum, otherRigidBodyNum=self.otherRigidBodyNum, bendingSensorNum=self.gripperNum, robotNum=self.robotNum)
        participantMotion = ParticipantMotion(defaultParticipantNum=self.participantNum, otherRigidBodyNum=self.otherRigidBodyNum, motionInputSystem=motionDataInputMode, mocapServer=self.motiveserverIpAddress, mocapLocal=self.motivelocalIpAddress, gripperInputSystem=gripperDataInputMode, bendingSensorNum=self.gripperNum, BendingSensor_ConnectionMethod="wired", bendingSensorUdpIpAddress=self.wirelessIpAddress, bendingSensorUdpPort=self.bendingSensorPorts, bendingSensorSerialCOMs=self.bendingSensorComs)

        # ----- Initialize robot arm ----- #
        if isEnablexArm:
            arm_1 = XArmAPI(self.xArmIpAddress_left)
            self.InitializeAll(arm_1, transform_left)

            arm_2 = XArmAPI(self.xArmIpAddress_right)
            self.InitializeAll(arm_2, transform_right)

        # ----- Control flags ----- #
        isMoving = False

        try:
            while True:
                if isMoving:
                    # ---------- Start control process timer ---------- #
                    loopStartTime = time.perf_counter()

                    # ----- Get transform data----- #
                    localPosition = participantMotion.LocalPosition(loopCount=self.loopCount)
                    localRotation = participantMotion.LocalRotation(loopCount=self.loopCount)

                    relativePosition = caMotion.GetRelativePosition(position=localPosition)
                    relativeRotation = caMotion.GetRelativeRotation(rotation=localRotation)

                    robotpos, robotrot = caMotion.participant2robot(relativePosition, relativeRotation, weightList)

                    if isEnablexArm:
                        # ----- Send to xArm ----- #
                        arm_1.set_servo_cartesian(transform_left.Transform(relativepos=robotpos["robot1"], relativerot=robotrot["robot1"], isLimit=False))
                        arm_2.set_servo_cartesian(transform_right.Transform(relativepos=robotpos["robot2"], relativerot=robotrot["robot2"], isLimit=False))

                    # ----- Bending sensor ----- #
                    dictGripperValue_R, dictGripperValue_P = (participantMotion.GripperControlValue(weight=weightGripperList, loopCount=self.loopCount))

                    # ----- Gripper control ----- #
                    if isEnablexArm:
                        code_1, ret_1 = arm_1.getset_tgpio_modbus_data(self.ConvertToModbusData(dictGripperValue_R["gripperValue1"]))
                        code_2, ret_2 = arm_2.getset_tgpio_modbus_data(self.ConvertToModbusData(dictGripperValue_R["gripperValue2"]))

                    # ----- Data recording ----- #
                    if self.isExportData:
                        dataRecordManager.Record(relativePosition, relativeRotation, weightList, dictGripperValue_P, robotpos, robotrot, dictGripperValue_R, time.perf_counter() - taskStartTime)

                    # ----- If xArm error has occured ----- #
                    if isEnablexArm and arm_1.has_err_warn:
                        isMoving = False
                        self.errorCount += 1
                        self.taskTime.append(time.perf_counter() - taskStartTime)
                        print('[ERROR] >> xArm Error has occured. Please enter "r" to reset xArm, or "q" to quit')

                    if isEnablexArm and arm_2.has_err_warn:
                        isMoving = False
                        self.errorCount += 1
                        self.taskTime.append(time.perf_counter() - taskStartTime)
                        print('[ERROR] >> xArm Error has occured. Please enter "r" to reset xArm, or "q" to quit')

                    # ---------- End control process timer ---------- #
                    processDuration = time.perf_counter() - loopStartTime

                    # ----- Fixed frame rate ----- #
                    if isFixedFrameRate:
                        sleepTime = loopTime - processDuration
                        if sleepTime < 0:
                            pass
                        else:
                            time.sleep(sleepTime)

                    self.loopCount += 1

                else:
                    keycode = input('Input > "q": quit, "r": Clean error and init arm, "s": start control \n')
                    # ----- Quit program ----- #
                    if keycode == "q":
                        if isEnablexArm:
                            arm_1.disconnect()
                            arm_2.disconnect()
                        self.PrintProcessInfo()

                        windll.winmm.timeEndPeriod(1)
                        break

                    # ----- Reset xArm and gripper ----- #
                    elif keycode == "r":
                        if isEnablexArm:
                            self.InitializeAll(arm_1, transform_left)
                            self.InitializeAll(arm_2, transform_right)
                            # self.InitRobotArm(arm, transform)
                            # self.InitGripper(arm)

                    # ----- Start streaming ----- #
                    elif keycode == "s":
                        caMotion.SetOriginPosition(participantMotion.LocalPosition())
                        caMotion.SetInversedMatrix(participantMotion.LocalRotation())

                        # ----- weight slider list ----- #
                        self.weightListPos[0].remove("weightListPos")
                        self.weightListRot[0].remove("weightListRot")
                        weightListPosstr = self.weightListPos[0]
                        weightListRotstr = self.weightListRot[0]
                        weightListPosfloat = list(map(float, weightListPosstr))
                        weightListRotfloat = list(map(float, weightListRotstr))
                        weightList = [weightListPosfloat,weightListRotfloat]

                        # ----- weight slider list ----- #
                        self.weightGripperList[0].remove("weightGripperList")
                        weightGripperListstr = self.weightGripperList[0]
                        weightGripperList = list(map(float, weightGripperListstr))

                        robotpos, robotrot = caMotion.participant2robot(participantMotion.LocalPosition(), participantMotion.LocalRotation(), weightList)
                        participantMotion.SetInitialBendingValue()

                        isMoving = True
                        taskStartTime = time.perf_counter()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt >> Stop: mainloop()")

            self.taskTime.append(time.perf_counter() - taskStartTime)
            self.PrintProcessInfo()

            if self.isExportData:
                dataRecordManager.ExportSelf(dirPath=self.dirPath, participant=self.participantname, conditions=self.condition, number=self.number)

            # ----- Disconnect ----- #
            if isEnablexArm:
                arm_1.disconnect()
                arm_2.disconnect()

            windll.winmm.timeEndPeriod(1)

        except:
            print("----- Exception has occurred -----")
            windll.winmm.timeEndPeriod(1)
            import traceback

            traceback.print_exc()

    def ConvertToModbusData(self, value: int):
        """
        Converts the data to modbus type.

        Parameters
        ----------
        value: int
            The data to be converted.
            Range: 0 ~ 800
        """

        if int(value) <= 255 and int(value) >= 0:
            dataHexThirdOrder = 0x00
            dataHexAdjustedValue = int(value)

        elif int(value) > 255 and int(value) <= 511:
            dataHexThirdOrder = 0x01
            dataHexAdjustedValue = int(value) - 256

        elif int(value) > 511 and int(value) <= 767:
            dataHexThirdOrder = 0x02
            dataHexAdjustedValue = int(value) - 512

        elif int(value) > 767 and int(value) <= 1123:
            dataHexThirdOrder = 0x03
            dataHexAdjustedValue = int(value) - 768

        modbus_data = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04, 0x00, 0x00]
        modbus_data.append(dataHexThirdOrder)
        modbus_data.append(dataHexAdjustedValue)

        return modbus_data

    def PrintProcessInfo(self):
        """
        Print process information.
        """

        print("----- Process info -----")
        print("Total loop count > ", self.loopCount)
        for ttask in self.taskTime:
            print("Task time\t > ", ttask, "[s]")
        print("Error count\t > ", self.errorCount)
        print("------------------------")

    def InitializeAll(self, robotArm, transform, isSetInitPosition=True):
        """
        Initialize the xArm

        Parameters
        ----------
        robotArm: XArmAPI
            XArmAPI object.
        transform: xArmTransform
            xArmTransform object.
        isSetInitPosition: (Optional) bool
            True -> Set to "INITIAL POSITION" of the xArm studio
            False -> Set to "ZERO POSITION" of the xArm studio
        """

        robotArm.connect()
        if robotArm.warn_code != 0:
            robotArm.clean_warn()
        if robotArm.error_code != 0:
            robotArm.clean_error()
        robotArm.motion_enable(enable=True)
        robotArm.set_mode(0)  # set mode: position control mode
        robotArm.set_state(state=0)  # set state: sport state
        if isSetInitPosition:
            initX, initY, initZ, initRoll, initPitch, initYaw = transform.GetInitialTransform()
            robotArm.set_position(x=initX, y=initY, z=initZ, roll=initRoll, pitch=initPitch, yaw=initYaw, wait=True)
        else:
            robotArm.reset(wait=True)
        print("Initialized > xArm")

        robotArm.set_tgpio_modbus_baudrate(2000000)
        robotArm.set_gripper_mode(0)
        robotArm.set_gripper_enable(True)
        robotArm.set_gripper_position(850, speed=5000)
        robotArm.getset_tgpio_modbus_data(self.ConvertToModbusData(425))
        print("Initialized > xArm gripper")

        robotArm.set_mode(1)
        robotArm.set_state(state=0)
