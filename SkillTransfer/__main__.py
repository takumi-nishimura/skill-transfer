from ctypes import windll

import RobotArmController.RobotControlManager

def main():
    robotControlManager = RobotArmController.RobotControlManager.RobotControlManagerClass()
    robotControlManager.mainloop()

if __name__ == "__main__":
    robotControlManager = (
        RobotArmController.RobotControlManager.RobotControlManagerClass()
    )
    robotControlManager.SendDataToRobot(

        isFixedFrameRate=False,
        isExportData=False,
        isEnablexArm=True,
    )

    print("\n----- End program -----")
