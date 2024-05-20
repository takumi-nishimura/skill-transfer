import socket
import inspect
import atexit

class UDPManager:
    data = None
    sock = None

    def __init__(self, port, localAddr, multicastAddr: str = '239.255.0.1', bufSize: int = 4096) -> None:
        """
        Initialize UDPManager

        Paramters
        ----------
        port: int
            Received UDP port
        localAddr: str
            Local IP address
        multicastAddr: (Optional) str
            Multicast IP address
        bufSize: (Optional) int
            Received buffer size
        """
        if port == 9000:
            print("hello")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.setsockopt(socket.IPPROTO_IP,
                        socket.IP_ADD_MEMBERSHIP,
                        socket.inet_aton(multicastAddr) + socket.inet_aton(localAddr))
        #self.sock.settimeout(timeOut)
        
        self.port               = port
        self.localAddress       = localAddr
        self.multicastAddress   = multicastAddr
        self.bufferSize         = bufSize

        #atexit.register(self.CloseSocket)

        # ----- Caller identification ----- #
        fileName = inspect.stack()[1].filename
        idx = fileName.find('Python')
        calleeFileName = fileName[idx+len('Python'):]

        print('UDPManager: Init UDP socket.     Called by ' + calleeFileName)

    def ReceiveData(self):
        """
        Receiving data once from the UDP socket.

        Returns
        ----------
        data: list[str]
            Receiving data from UDP socket.
        cli_addr: str
            Sender address.
        """
        
        data, cli_addr = self.sock.recvfrom(self.bufferSize)
        self.data = data.decode(encoding='utf-8').split(',')
        print(self.data)

        return self.data, cli_addr
    
    def UpdateData(self) -> None:
        """
        Receives data from the UDP stream and keeps updating the self.data.
        """

        while True:
            data, cli_addr = self.sock.recvfrom(self.bufferSize)
            self.data = data.decode(encoding='utf-8').split(',')
    
    def SendData(self, data, targetAddr, targetPort):
        """
        Send data to the UDP socket.
        TODO: Operation not verified
        """

        send_len = self.sock.sendto(str(data).encode('utf-8'), (targetAddr, targetPort))
        
        # ----- REGACY: Hagi ----- #
        #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_address))

        #send_len = self.sock.sendto(data.encode('utf-8'), (targetAddr, targetPort))
    
    def CloseSocket(self) -> None:
        """
        Closing the UDP socket.
        """
        
        self.sock.close()
        print('UDPManager: Closing socket')