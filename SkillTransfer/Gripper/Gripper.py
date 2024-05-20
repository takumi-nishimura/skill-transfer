import serial
from Gripper.UDP import UDPManager


class BendingSensorManager:

    bendingValue = 0

    def __init__(self, BendingSensor_connectionmethod, ip, port) -> None:
        self.ip = ip
        self.port = port
        self.bufsize = 4096
        self.bendingValue = 425

        if BendingSensor_connectionmethod == "wireless":
            self.udpManager = UDPManager(port=self.port, localAddr=self.ip)

        elif BendingSensor_connectionmethod == "wired":
            self.serialObject = serial.Serial(ip, port)
            # not_used = self.serialObject.readline()

    def StartReceiving(self, fromUdp: bool = False):
        """
        Receiving data from bending sensor and update self.bendingValue
        """

        if fromUdp:
            sock = self.udpManager.sock

            try:
                while True:
                    data, addr = self.udpManager.ReceiveData()
                    self.bendingValue = float(data[0])

            except OSError:
                print(
                    "[OSError] UDPManager >> I/O related errors. Please check the UDP socket."
                )

            except KeyboardInterrupt:
                print("KeyboardInterrupt >> Stop: BendingSensorManager.py")

        else:
            try:
                while True:
                    data = self.serialObject.readline()
                    self.bendingValue = float(data.strip().decode("utf-8"))

            except KeyboardInterrupt:
                print("KeyboardInterrupt >> Stop: BendingSensorManager.py")

    def EndReceiving(self):
        self.udpManager.CloseSocket()
