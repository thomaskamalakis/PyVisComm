# PyVisComm
We have implemented two basic components of PyVisComm. 
The one is libdmt which is used to simulate the DMT transceiver in great detail. 
The second component is libimres which can be used to calculate the impulse response of the optical wireless channel.

This project contains two modules. File description:

1. libdmt module
The libdmt module implements auxiliary functions and the dmtphy class which is the central class associated 
with the physical layer of the DMT transmitter. The dmtphy class defines a set of attributes which reflect the parameters of the system. 

2. libimres module
The libimres module is used to calculate the impulse response of an optical wireless channel. 
