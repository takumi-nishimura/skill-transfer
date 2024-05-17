from math import pi
import numpy as np
from MotionFilter.MotionFilter import MotionFilter
from FileIO.FileIO import FileIO


class xArmTransform_right:
    """
    xArmの座標と回転を保持するクラス
    """

    x, y, z = 0, 0, 0
    roll, pitch, yaw = 0, 0, 0

    # ----- Initial Position and Rotation ----- #
    __initX, __initY, __initZ = 300, -200, 150
    __initRoll, __initPitch, __initYaw = 180, 45, 90

    # ----- Minimum limitation ----- #
    __minX, __minY, __minZ = 310, -300, 225
    __minRoll, __minPitch, __minYaw = -90, -65, -90

    # ----- Maximum limitation ----- #
    __maxX, __maxY, __maxZ = 650, 300, 650
    __maxRoll, __maxPitch, __maxYaw = 90, 70, 90

    def __init__(self):
        self.n = 4
        self.fp = 10
        self.fs = 180
        self.filter_robot = MotionFilter()
        self.filter_robot.InitLowPassFilterWithOrder(self.fs, self.fp, self.n)

        self.beforefilt = [[0, 0, 0, 0, 0, 0]] * self.n
        self.afterfilt = [[0, 0, 0, 0, 0, 0]] * self.n

        fileIO = FileIO()

        dat = fileIO.Read("settings.csv", ",")
        xArmIP1 = [addr for addr in dat if "xArmIPAddress1" in addr[0]][0][1]
        xArmIP2 = [addr for addr in dat if "xArmIPAddress2" in addr[0]][0][1]
        self.xArmIpAddress2 = xArmIP2

    def SetInitialTransform(self, initX, initY, initZ, initRoll, initPitch, initYaw):
        """
        Set the initial position and rotation.
        If this function is not called after the class is instantiated, the initial values of the member variables of this class will be used.

        Parameters
        ----------
        initX, initY, initZ: float
            Initial position.
        initRoll, initPitch, initYaw: float
            Initial rotation.
        """

        self.__initX = initX
        self.__initY = initY
        self.__initZ = initZ

        self.__initRoll = initRoll
        self.__initPitch = initPitch
        self.__initYaw = initYaw

    def GetInitialTransform(self):
        """
        Get the initial position and rotation.
        """

        return (
            self.__initX,
            self.__initY,
            self.__initZ,
            self.__initRoll,
            self.__initPitch,
            self.__initYaw,
        )

    def SetMinimumLimitation(self, minX, minY, minZ, minRoll, minPitch, minYaw):
        """
        Set the lower limit of the position and rotation.
        If this function is not called after the class is instantiated, the initial values of the member variables of this class will be used.

        Parameters
        ----------
        minX, minY, minZ: float
            Lower limit of the position.
        minRoll, minPitch, minYaw: float
            Lower limit of the rotation.
        """

        self.__minX = minX
        self.__minY = minY
        self.__minZ = minZ

        self.__minRoll = minRoll
        self.__minPitch = minPitch
        self.__minYaw = minYaw

    def SetMaximumLimitation(self, maxX, maxY, maxZ, maxRoll, maxPitch, maxYaw):
        """
        Set the upper limit of the position and rotation.
        If this function is not called after the class is instantiated, the initial values of the member variables of this class will be used.

        Parameters
        ----------
        maxX, maxY, maxZ: float
            Upper limit of the position.
        maxRoll, maxPitch, maxYaw: float
            Upper limit of the rotation.
        """

        self.__maxX = maxX
        self.__maxY = maxY
        self.__maxZ = maxZ

        self.__maxRoll = maxRoll
        self.__maxPitch = maxPitch
        self.__maxYaw = maxYaw

    def Transform(
        self, posMagnification=1, rotMagnification=1, isLimit=True, isOnlyPosition=False
    ):
        """
        Calculate the position and rotation to be sent to xArm.

        Parameters
        ----------
        posMagnification: int (Default = 1)
            Magnification of the position. Used when you want to move the position less or more.
        rotMagnification: int (Default = 1)
            Magnification of the rotation. Used when you want to move the position less or more.
        isLimit: bool (Default = True)
            Limit the position and rotation.
            Note that if it is False, it may result in dangerous behavior.
        isOnlyPosition: bool (Default = True)
            Reflect only the position.
            If True, the rotations are __initRoll, __initPitch, and __initYaw.
            If False, the rotation is also reflected.
        """

        x, y, z = (
            self.x * posMagnification + self.__initX,
            self.y * posMagnification + self.__initY,
            self.z * posMagnification + self.__initZ,
        )
        roll, pitch, yaw = (
            self.roll * rotMagnification + self.__initRoll,
            self.pitch * rotMagnification + self.__initPitch,
            self.yaw * rotMagnification + self.__initYaw,
        )

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            # pos X
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            # pos Y
            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            # pos Z
            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            # Roll
            if 0 < roll < self.__maxRoll:
                print("roll1")
                roll = self.__maxRoll
            elif self.__minRoll < roll < 0:
                print("roll2")
                roll = self.__minRoll

            # Pitch
            if pitch > self.__maxPitch:
                print("pitch1")
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                print("pitch2")
                pitch = self.__minPitch

            # Yaw
            if yaw > self.__maxYaw:
                print("yaw1")
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                print("yaw2")
                yaw = self.__minYaw

            print([x, y, z, roll, pitch, yaw])

        return np.array([x, y, z, roll, pitch, yaw])

    def TransformwithLPF(
        self, posMagnification=1, rotMagnification=1, isLimit=True, isOnlyPosition=True
    ):
        """
        Calculate the position and rotation to be sent to xArm.

        Parameters
        ----------
        posMagnification: int (Default = 1)
            Magnification of the position. Used when you want to move the position less or more.
        rotMagnification: int (Default = 1)
            Magnification of the rotation. Used when you want to move the position less or more.
        isLimit: bool (Default = True)
            Limit the position and rotation.
            Note that if it is False, it may result in dangerous behavior.
        isOnlyPosition: bool (Default = True)
            Reflect only the position.
            If True, the rotations are __initRoll, __initPitch, and __initYaw.
            If False, the rotation is also reflected.
        """

        self.beforefilt.append(
            [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
        )
        self.posfilt = self.filter_robot.lowpass2(self.beforefilt, self.afterfilt)
        self.afterfilt.append(self.posfilt)
        del self.beforefilt[0]
        del self.afterfilt[0]

        x, y, z = (
            self.posfilt[0] * posMagnification + self.__initX,
            self.posfilt[1] * posMagnification + self.__initY,
            self.posfilt[2] * posMagnification + self.__initZ,
        )
        roll, pitch, yaw = (
            self.posfilt[3] * rotMagnification + self.__initRoll,
            self.posfilt[4] * rotMagnification + self.__initPitch,
            self.posfilt[5] * rotMagnification + self.__initYaw,
        )

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            # pos X
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            # pos Y
            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            # pos Z
            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            # Roll
            if 0 < roll < self.__maxRoll:
                roll = self.__maxRoll
            elif self.__minRoll < roll < 0:
                roll = self.__minRoll

            # Pitch
            if pitch > self.__maxPitch:
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                pitch = self.__minPitch

            # Yaw
            if yaw > self.__maxYaw:
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                yaw = self.__minYaw

        return np.array([x, y, z, roll, pitch, yaw])


class xArmTransform_left:
    """
    xArmの座標と回転を保持するクラス
    """

    x, y, z = 0, 0, 0
    roll, pitch, yaw = 0, 0, 0

    # ----- Initial Position and Rotation ----- #
    __initX, __initY, __initZ = 300, 200, 150
    __initRoll, __initPitch, __initYaw = 180, -45, 90

    # ----- Minimum limitation ----- #
    __minX, __minY, __minZ = 310, -300, 225
    __minRoll, __minPitch, __minYaw = -90, -65, -90

    # ----- Maximum limitation ----- #
    __maxX, __maxY, __maxZ = 650, 300, 650
    __maxRoll, __maxPitch, __maxYaw = 90, 70, 90

    def __init__(self):
        self.n = 4
        self.fp = 10
        self.fs = 180
        self.filter_robot = MotionFilter()
        self.filter_robot.InitLowPassFilterWithOrder(self.fs, self.fp, self.n)

        self.beforefilt = [[0, 0, 0, 0, 0, 0]] * self.n
        self.afterfilt = [[0, 0, 0, 0, 0, 0]] * self.n

    def SetInitialTransform(self, initX, initY, initZ, initRoll, initPitch, initYaw):
        """
        Set the initial position and rotation.
        If this function is not called after the class is instantiated, the initial values of the member variables of this class will be used.

        Parameters
        ----------
        initX, initY, initZ: float
            Initial position.
        initRoll, initPitch, initYaw: float
            Initial rotation.
        """

        self.__initX = initX
        self.__initY = initY
        self.__initZ = initZ

        self.__initRoll = initRoll
        self.__initPitch = initPitch
        self.__initYaw = initYaw

    def GetInitialTransform(self):
        """
        Get the initial position and rotation.
        """

        return (
            self.__initX,
            self.__initY,
            self.__initZ,
            self.__initRoll,
            self.__initPitch,
            self.__initYaw,
        )

    def SetMinimumLimitation(self, minX, minY, minZ, minRoll, minPitch, minYaw):
        """
        Set the lower limit of the position and rotation.
        If this function is not called after the class is instantiated, the initial values of the member variables of this class will be used.

        Parameters
        ----------
        minX, minY, minZ: float
            Lower limit of the position.
        minRoll, minPitch, minYaw: float
            Lower limit of the rotation.
        """

        self.__minX = minX
        self.__minY = minY
        self.__minZ = minZ

        self.__minRoll = minRoll
        self.__minPitch = minPitch
        self.__minYaw = minYaw

    def SetMaximumLimitation(self, maxX, maxY, maxZ, maxRoll, maxPitch, maxYaw):
        """
        Set the upper limit of the position and rotation.
        If this function is not called after the class is instantiated, the initial values of the member variables of this class will be used.

        Parameters
        ----------
        maxX, maxY, maxZ: float
            Upper limit of the position.
        maxRoll, maxPitch, maxYaw: float
            Upper limit of the rotation.
        """

        self.__maxX = maxX
        self.__maxY = maxY
        self.__maxZ = maxZ

        self.__maxRoll = maxRoll
        self.__maxPitch = maxPitch
        self.__maxYaw = maxYaw

    def Transform(
        self, posMagnification=1, rotMagnification=1, isLimit=True, isOnlyPosition=False
    ):
        """
        Calculate the position and rotation to be sent to xArm.

        Parameters
        ----------
        posMagnification: int (Default = 1)
            Magnification of the position. Used when you want to move the position less or more.
        rotMagnification: int (Default = 1)
            Magnification of the rotation. Used when you want to move the position less or more.
        isLimit: bool (Default = True)
            Limit the position and rotation.
            Note that if it is False, it may result in dangerous behavior.
        isOnlyPosition: bool (Default = True)
            Reflect only the position.
            If True, the rotations are __initRoll, __initPitch, and __initYaw.
            If False, the rotation is also reflected.
        """

        x, y, z = (
            self.x * posMagnification + self.__initX,
            self.y * posMagnification + self.__initY,
            self.z * posMagnification + self.__initZ,
        )
        roll, pitch, yaw = (
            self.roll * rotMagnification + self.__initRoll,
            self.pitch * rotMagnification + self.__initPitch,
            self.yaw * rotMagnification + self.__initYaw,
        )

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            # pos X
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            # pos Y
            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            # pos Z
            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            # Roll
            if 0 < roll < self.__maxRoll:
                print("roll1")
                roll = self.__maxRoll
            elif self.__minRoll < roll < 0:
                print("roll2")
                roll = self.__minRoll

            # Pitch
            if pitch > self.__maxPitch:
                print("pitch1")
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                print("pitch2")
                pitch = self.__minPitch

            # Yaw
            if yaw > self.__maxYaw:
                print("yaw1")
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                print("yaw2")
                yaw = self.__minYaw

            print([x, y, z, roll, pitch, yaw])

        return np.array([x, y, z, roll, pitch, yaw])

    def TransformwithLPF(
        self, posMagnification=1, rotMagnification=1, isLimit=True, isOnlyPosition=True
    ):
        """
        Calculate the position and rotation to be sent to xArm.

        Parameters
        ----------
        posMagnification: int (Default = 1)
            Magnification of the position. Used when you want to move the position less or more.
        rotMagnification: int (Default = 1)
            Magnification of the rotation. Used when you want to move the position less or more.
        isLimit: bool (Default = True)
            Limit the position and rotation.
            Note that if it is False, it may result in dangerous behavior.
        isOnlyPosition: bool (Default = True)
            Reflect only the position.
            If True, the rotations are __initRoll, __initPitch, and __initYaw.
            If False, the rotation is also reflected.
        """

        self.beforefilt.append(
            [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
        )
        self.posfilt = self.filter_robot.lowpass2(self.beforefilt, self.afterfilt)
        self.afterfilt.append(self.posfilt)
        del self.beforefilt[0]
        del self.afterfilt[0]

        x, y, z = (
            self.posfilt[0] * posMagnification + self.__initX,
            self.posfilt[1] * posMagnification + self.__initY,
            self.posfilt[2] * posMagnification + self.__initZ,
        )
        roll, pitch, yaw = (
            self.posfilt[3] * rotMagnification + self.__initRoll,
            self.posfilt[4] * rotMagnification + self.__initPitch,
            self.posfilt[5] * rotMagnification + self.__initYaw,
        )

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            # pos X
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            # pos Y
            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            # pos Z
            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            # Roll
            if 0 < roll < self.__maxRoll:
                roll = self.__maxRoll
            elif self.__minRoll < roll < 0:
                roll = self.__minRoll

            # Pitch
            if pitch > self.__maxPitch:
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                pitch = self.__minPitch

            # Yaw
            if yaw > self.__maxYaw:
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                yaw = self.__minYaw

        return np.array([x, y, z, roll, pitch, yaw])
