#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:24:38 2019

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy.polynomial.polynomial import polyval
import libconstants as const
import time
import random 

# exponential response function - used for testing
def expres(a,t):

    x = np.zeros(t.size)
    i = np.where(t >= 0)
    x[i] = a*np.exp(-a*t[i])
    return(x)

def calcfreqaxis(t):
# calculate frequency axis

    Dt = t[1]-t[0]
    Nt = t.size
    Dfs = 1.0/(Nt*Dt)
    freqaxis = np.arange( -Nt/2.0, Nt/2.0,  1.0) * Dfs
    return(freqaxis)

def rms(x):
    """
    Calculate RMS value of signal
    """
    S=np.sum(np.abs(x)**2.0) / x.size
    return np.sqrt(S)
    
# analog Fourier transform via FFT
def spec(t,x):

    Dt = t[1]-t[0]
    Nt = t.size
    Df = 1.0/(Nt*Dt)
    f = np.arange( -Nt/2.0, Nt/2.0,  1.0) * Df
    X = Dt * np.fft.fftshift( np.fft.fft (np.fft.fftshift(x) ))
    return f,X

# inverse analog Fourier transfrom via IFFT
def invspec(f,X):

    Df = f[1]-f[0]
    Nf = f.size
    Dt = 1.0/(Nf*Df)
    t = np.arange( -Nf/2.0, Nf/2.0,  1.0) * Dt
    x = Nf * Df * np.fft.fftshift( np.fft.ifft (np.fft.fftshift(X) ))
    return t,x

# convert digital signal to analog
def converttoanalog(t,din,Ts,t0=0.0,gfilterbandwidth=None):
    t=t-t0
    m=np.round( t/Ts ).astype(int)
    N=din.size
    x=np.zeros(t.size)
    i=np.where( (m>=0) & (m < N) )
    x[i]=din[m[i]]
    if gfilterbandwidth!=None:
        f,P=spec(t,x)
        H=np.exp(-f**2.0/2/gfilterbandwidth**2)
        Q=P*H
        _,x=invspec(f,Q)
    return(x)

# sample analog waveform
def sample(t,x,Ts,toffset=0.0,tinitial=None,tduration=None):

    if tinitial == None:
        tinitial = np.min(t)

    if tduration == None:
        tduration = np.max(t) - np.min(t)

    # find time instances within the specified interval
    ts = t[ (t>=tinitial) & (t<tinitial + tduration) ]

    # subtract to set the first time instance at t=0
    ts = ts - tinitial

    # obtain the corresponding values of the analog waveform
    xs=  x[ (t>=tinitial) & (t<tinitial + tduration) ]

    # find in which sample duration the values of the time axis correspond
    m = np.floor( ts/Ts ).astype(int)

    # sampling times
    tout = m*Ts
    tout = np.unique(tout) + toffset

    # sample by interpolation
    dout = np.interp(tout,ts,xs)

    # remember to reset the time axis    
    # check wheter we exceed the maximum duration
    dout = dout[(tout >= tinitial) & (tout < tinitial + tduration)]
    tout = tout[(tout >= tinitial) & (tout < tinitial + tduration)]
    
    return(tout,dout)

# provide complex conjugate symmetry so that the IFFT is real
def addconjugates(din):

    N=din.size

    # ensure DC component is real
    din[0]=np.real(din[0])

    # calculate conjugate block
    conjblock=np.flip(np.conj(din[1:]))

    # new block to contain the conjugates
    dout=np.zeros(2*N) + 1j * np.zeros(2*N)

    # original part
    dout[0:N]=din

    # conjugate part
    dout[N+1:]=conjblock

    # Nth component must be real
    dout[N]=din[0]

    return(dout)

# Generate bit sequences for gray code of order M
def graycode(M):

    if (M==1):
        g=['0','1']

    elif (M>1):
        gs=graycode(M-1)
        gsr=gs[::-1]
        gs0=['0'+x for x in gs]
        gs1=['1'+x for x in gsr]
        g=gs0+gs1
    return(g)

# convert stream of bits to bit blocks of size Mi. If Mi is a numpy array the process is repeated cyclically.
def bitblockscyc(b,Mi):

    blocks=[]

    fullrepetitions=0
    curr=0
    bitsleft=b

    while len(bitsleft) >= Mi[curr]:
        currbits=bitsleft[0:Mi[curr]]
        bitsleft=bitsleft[Mi[curr]:]
        blocks.append(currbits)
        curr=curr+1
        if curr>=Mi.size:
            curr=0
            fullrepetitions=fullrepetitions+1

    return blocks,bitsleft,fullrepetitions

# convert stream of bits to bit blocks of size Mi. If Mi is a numpy array the process is repeated cyclically. Blocks are arranged in two dimensions
def bitblockscyc2D(b,Mi):

    blocks=[]

    # initialize empty blocks for each value of Mi
    for mi in Mi:
        blocks.append([])

    fullrepetitions=0
    curr=0
    bitsleft=b

    while len(bitsleft) >= Mi[curr]:
        currbits=bitsleft[0:Mi[curr]]
        bitsleft=bitsleft[Mi[curr]:]
        blocks[curr].append(currbits)
        curr=curr+1
        if curr>=Mi.size:
            curr=0
            fullrepetitions=fullrepetitions+1

    return blocks,bitsleft,fullrepetitions

def counterrors(b1,b2):
    """
    Count errors between bit sequences b1 and b2
    """
    b1=bitstrtobits(b1)
    b2=bitstrtobits(b2)
    
    diff = np.abs(b1-b2)
    errors=np.sum(diff).astype(int)
    return(errors)

def bitstrblockstobitstr(blocks):    
    return ''.join(blocks)
    
# convert stream of bits to bit blocks of size Mi. If Mi is a numpy array the process is NOT repeated cyclically!!!
def bitblocks(b,Mi):

    blocks=[]

    curr=0
    bitsleft=b
    toread=Mi[curr]

    while len(bitsleft) >= toread:
        currbits=bitsleft[0:Mi[curr]]
        bitsleft=bitsleft[Mi[curr]:]
        blocks.append(currbits)
        curr=curr+1
        if (curr<Mi.size):
           toread=Mi[curr]
        else:
           break
    return blocks,bitsleft,curr

# convert a set of np.array bits to bit string
def bitstobitstr(b):
    bitstr=''
    for bi in b:
        bitstr=bitstr+str(bi)

    return(bitstr)

 # convert a bit string to an np.array
def bitstrtobits(b):
    bits=np.zeros(len(b))
    for i,v in enumerate(b):
        bits[i]=int(v)

    return(bits)

# plot bits
def visualizebitblock(bitsb,zoomfrom=None,zoomto=None):

    fig=plt.figure()
    start=1
    marker='ro'
    color='r'

    if isinstance(bitsb,str):
        bitsb=[bitsb]

    for b in bitsb:
        bits=bitstrtobits(b)
        end=start+bits.size
        x=np.arange(start,end)

        plt.stem(x,bits,linefmt=color,markerfmt=marker,use_line_collection=True,basefmt=" ")
        if marker=='ro':
            marker='bo'
            color='b'
        else:
            marker='ro'
            color='r'
        start=end

    if zoomfrom!=None:
        start=zoomfrom
    else:
        start=1

    if zoomto!=None:
        end=zoomto

    plt.xlim([start,end])

# PAM symbol dictionary
def pamsymbols(M):
    m=np.arange(0,M)
    symbols=2*m-M+1
    return(symbols)

# PAM symbol at index m
def pamsymbol(m,M):
    return(2*m-M+1)

def qammapeven(order=16):
    """
    QAM Constellation for order = 2^(2n)
    """
    m = np.log2(order).astype(int)
    Ms = np.sqrt(order)
    gc = graycode( m/2 )
    forward = {}                                        # bits to symbols
    backward = np.zeros(order) + 1j * np.zeros(order)    
    for i,gi in enumerate(gc):
       for j,gj in enumerate(gc):
           q = np.complex(pamsymbol(i,Ms),pamsymbol(j,Ms))
           forward[gi+gj] = q
           indx = int( gi+gj , 2 )
           backward[indx] = q
    
    return forward, backward
           
def qammapodd(order=32):
    """
    Map bit to QAM symbols for M=2^(2n+1) orderings
    """
    forward = {}                                        # bits to symbols
    backward = np.zeros(order) + 1j * np.zeros(order) 
    
    m = np.log2(order).astype(int)
    
    if  m % 2 == 1:
    
        l = (m-1)/2+1
        s = (m-1)/2
        
        l = l.astype(int)
        Gl = graycode( l )
        Gs = graycode( s )
        
        n = ((m-1) / 2).astype(int)
            
        # Start from a (m+1) x m configuration
        Q = np.zeros([2**n,2**(n+1)]) + 1j * np.zeros([2**n,2**(n+1)])
        bits = []
          
        for my in range(0,2**n):
          B = []
          for mx in range(0,2**(n+1)):
            Q[my,mx] = (2**(n+1) - 2*mx - 1) +1j * (2**n - 2*my - 1)
            B.append( Gl[mx] + Gs[my])
          bits.append(B)   
            
    # Transform constellation
    s = 2 ** ( s-1 )
    
    for my in range(0,2**n):
       for mx in range(0,2**(n+1)):
           
           q=Q[my,mx]
           b=bits[my][mx]
           
           irct = np.real( q )
           qrct = np.imag( q )
           
           if np.abs( irct ) < 3 * s:
               i = irct
               q = qrct
           elif np.abs(np.imag(q)) > s:
               i = np.sign( irct ) * (np.abs(irct) - 2*s)
               q = np.sign( qrct ) * (4*s - np.abs(qrct))
           else: 
               i = np.sign( irct ) * (4*s - np.abs(irct))
               q = np.sign( qrct ) * (np.abs(qrct) + 2*s)
           forward[b] = i + 1j *q
           indx = int( b , 2 )
           backward[indx] = forward[b]
    
    return forward, backward
           
        
        
def qammap(order=16):
    """
    Map bits to QAM symbols
    """
    
    
    m = np.log2(order).astype(int)
    
    # is this a rectangular shaped QAM ?    
    if  m % 2 == 0:
        forward,backward = qammapeven(order=order)
    else:
        forward,backward = qammapodd(order=order)
        
    avgpower = np.mean( np.abs (backward) ** 2.0 )        
    forwardn = {}
    backwardn = np.zeros(order) + 1j * np.zeros(order)
        
    s = np.sqrt(avgpower)
    for x in forward:
        forwardn[x] = forward[x] / s
            
    backwardn = backward / s
    
    return forward,backward,forwardn,backwardn,s


def findclosestanddecode(s,backwardmap):
    """
    Find closest symbol and decode
    """
    N = np.log2(backwardmap.size).astype(int)
    
    p = np.abs(backwardmap - s).argmin()
    sc = backwardmap[p]
    b = np.binary_repr(p,N)
    
    return sc, b         
           
# add cp to symbol sequence
def addcp(s,cplength):
    last=s.size
    start=last-cplength
    scp=np.concatenate((s[start:last],s))
    return(scp)

"""
Shortcut for converting an element
"""
def makelist(arg,N):
    if not(isinstance(arg,list)):
        return([arg] * N)
    else:
        return(arg)
"""
DMT physical layer class
"""

def noise(t,f=None,psd=None):
    """ 
    Add colored or white noise at the receiver
    """

    if psd is None:
        psd = lambda x: 1
    
    if not callable(psd):
        psd = lambda x: np.interp(x,f,psd)
    
    f = calcfreqaxis(t)
    H = psd(f)    
    Hf = np.sqrt(H)    
    r = np.random.randn(t.size)
    R = np.fft.fft(r)
    R = R * Hf
    x = np.fft.fftshift( np.fft.ifft( np.fft.fftshift(R) ) )
    
    return( np.real(x) )

class dmtphy:

    class timings:
        total = 0.0
        exectimes = {}
    
    def __init__(self,
                 nocarriers=16,                 # number of subcarrier channels
                 M=64,                          # QAM order
                 noframes=16,                   # frames to be considered
                 Df=1e6,                        # subcarrier spacing
                 cpsize=5,                      # size of CP prefix
                 samplespersymbol=40,           # samples per symbol duration
                 tapsize=20,                    # tapsize for channel coefficients
                 psd = None,                    # noise psd,
                 trfcallable = None,
                 sampleoffset=0,                # sampling offset at the receiver.
                                                # 0 indicates no offset
                                                # 0.5 Ts/2 offset
                                                # 1 Ts offset
                 scales = None,                 # power scales for the carriers, sum of squares must add up to nocarriers
                 cliplevel = None,              # clipping ratio                 
                 dacfilterbandwidth=None,       # filter of the DAC
                 polynl = np.array([0, 1])      # nonlinearity polynomial coefficients     
                 ):

        self.debug = False                            #change to true if we require debuging
        self.timecode = False                         #change to true if you want to time the execution

        '''
        Transmitter characteristics
        '''
        
        self.ommitzero = True                         # ignore the zeroth subcarrier
        
        if isinstance(M,int) or isinstance(M,float):
            M = M * np.ones(nocarriers)
        
        self.M = M.astype(int)                        # Modulation order for each subcarrier
        self.bin = ''                                 # input bits
        self.cpsize = cpsize                          # size of cyclic prefix
        self.noframes = int(2* round(noframes /2))    # number of DMT frames - must be an even number
        self.nocarriers = nocarriers                  # number of carriers
        self.t0 = 0                                   # time in the analog time axis where we assume that the
                                                      # frames start being transmitted
        self.samplespersymbol = samplespersymbol      # samples per symbol in the analog waveform
        self.dacfilterbandwidth = None                # filter bandwidth at the output of the DAC
        self.framesbefore = 20                        # guard period before frame transmission (empty frames)
        self.framesafter = 20                         # guard period after frame transmission (empty frames)
        self.carriermodulation = 'qam'                # modulation in carriers
        self.forwardmaps = None                       # forward symbol map for carriers bits -->> symbols
        self.backwardmaps = None                      # backward symbol map for carriers symbols -->> bits
        self.forwardmaps = None                       # normalized forward symbol map for carriers bits -->> symbols
        self.backwardmaps = None                      # normalized backward symbol map for carriers symbols -->> bits        
        self.sic = None                               # symbols assigned to carriers
        self.sicun = None                             # unscaled symbols assigned to carriers
        self.bic = None                               # bits assigned to carriers
        self.txframeswithcp = None                    # frames at TX with CP
        self.txs = None                               # output symbol sequence fed at the transmitter DAC
        self.txsunclipped = None                      # unclipped waveform samples at the DAC output
        self.analogtx = None                          # analog waveform at TX ourput
        self.scales = scales
        self.Df = Df                                  # subcarrier spacing
        self.Ts = 1.0/Df/(2.0*nocarriers)             # sample duration
        self.txinputifftframes = None                 # input blocks at IFFT input
        self.txoutputifftframes = None                # output blocks at IFFT input
        self.removeimags = True                       # removes imaginary parts from IFFT output
        self.framesamples = cpsize+2*nocarriers       # samples per frame
        self.Tframe = (cpsize+2*nocarriers)*self.Ts   # duration of the DMT frames
        self.Tsignal = self.Tframe*self.noframes      # duration of the DMT signal (without guard periods)
        self.anarlogt = None                          # analog time
        self.centertimeaxis = True                    # Center the analog time axis
        self.analogtx = None                          # analog waveform at the output of the transmitter
        self.analogtxspec = None                      # analog spectrum at the output of the transmitter
        
        if scales is None:
            self.scales = np.ones( self.nocarriers )
        else:
            self.scales = scales / np.sum( scales ) * self.nocarriers
        
        if dacfilterbandwidth is None:
            self.dacfilterbandwidth = 3.0/self.Ts
        else:
            self.dacfilterbandwidth = dacfilterbandwidth
        
        self.normalizesymbols = True                  # normalize symbols so that the average energy is equal to one?
        self.scalesforcarriers = None                 # scales required for symbol normalization
        self.crestfactor = None                       # Crest factor of the transmitter samples       
        self.nobits = None                            # number of bits to be transmitted
        self.cliplevel = cliplevel                    # Clipping level in dB
        self.Aclip = None                             # Amplitude corresponding to the clipping level
        self.DC = None                                # DC level at the transmitter
        self.Amax = None                              # specified maximum signal amplitude at the transmitter after adding DC component
        self.polynl = polynl                          # nonlinearity polynomial
        
        '''
        Channel characteristics
        '''

        self.taps=None                              # digital channel taps
        self.tapssamplesperTs=100                   # samples per sample duration when calculating the taps
        self.tapsguardperiod=20                     # defines guard period when calculating the taps
        self.freqaxis=None                          # frequency axis
        self.trf=None                               # transfer functon of the channel
        self.ht=None                                # analog channel impulse response
        self.tapsize=tapsize                        # number of taps
        self.trfcallable = trfcallable              # callable function for the transfer function of the channel
        self.psd = psd                              # noise psd to be added at the receiver input
        
        '''
        Receiver Characteristics
        '''
        self.analogrx=None                          # analog waveform at the receiver input
        self.analogrxspec=None                      # analog spectrum at the input of the receiver
        self.rxs=None                               # samples at the input of the receiver
        self.toffset=sampleoffset*self.Ts           # time offset for sampling at the receiver
        self.ts=None                                # times in which the analog receiver signal is sampled
        self.rxsd=None                              # the samples at the input of the receiver calculated using the digital channel approach
        self.rxframeswithcp=None                    # received DMT frames containing the cyclic prefix
        self.rxinputfftframes=None                  # received DMT frames without the cyclic prefix
        self.rxoutputfftframes=None                 # frames at the output of the FFT block
        self.rxsic=None                             # symbols assigned to carriers
        self.eqtaps=None                            # equalization taps. If None then simply use the inverse of the channel taps in the frequency domain
        self.rxsic=None                             # symbols obtained at RX subcarrier channels
        self.rxsicun=None                           # unscaled symbol estimates (original constellation)
        self.siest=None                             # symbol estimates after hard decoding
        self.rxbic=None                             # bit estimates at subcarriers
        self.bout=None                              # bits obtained at the output of the receiver
        self.berrors=None                           # bit errors
        self.berrorsinc=None                        # bit errors in carrier channels
        self.snr=None                               # Receive SNR at the various carrier channels 
        '''
        Simulation Sequences
        '''

        
        self.seqdig = ['setrandombits','setsymbolmaps','setcarriersymbols','calcifftinput',
                  'calcifftoutput','calccptxframes','calctxsymbols',
                  'cliptxsamples','normalizetxs','makedc','applytxnl','calctaps',                  
                  'applydigitalchannel','normalizerxs','calcrxframes','removeDC',
                  'removecprxframes','calcfftoutput','calcrxcarriersamples',
                  'calcrxestimates','calcrxbits','calcerrors','calcsnrevm','calcber'
                  ]
        
        self.seqanl = ['setrandombits','setsymbolmaps','setcarriersymbols','calcifftinput',
                  'calcifftoutput','calccptxframes','calctxsymbols',
                  'cliptxsamples','normalizetxs','makedc','applytxnl','calctaps','calctxwaveform','setchanneltrf',                  
                  'applyanalogchannel','calcadcoutput','removeDC','calcrxframes',
                  'removecprxframes','calcfftoutput','calcrxcarriersamples',
                  'calcrxestimates','calcrxbits','calcerrors','calcsnrevm'
                  ]
        
    # define the set of input bits, argument is a np array
    def setinputbits(self,bi):
        self.bin=bitstobitstr(bi)
        
    # define the set of input bits, argument is a bit string
    def setinputbitstr(self,bistr):
        self.bin=bistr
        
    def calcnumberofbits(self):
        """
        Calculate number of bits to be transmitted
        """ 
        # do we exclude the zeroth subcarrier?
        if self.ommitzero:
            bitsperframe = sum(np.log2(self.M[1:]).astype(int))
        else:
            bitsperframe = sum(np.log2(self.M).astype(int))

        Nbits=bitsperframe*self.noframes
        self.nobits = Nbits
        
    # assign random bits corresponding to the required frames
    def setrandombits(self):
        
        self.calcnumberofbits()
        bstr = ''.join(random.choice(['0','1']) for i in range(self.nobits))
        self.setinputbitstr(bstr)
        self.datarate = self.nobits / self.Tsignal

    # set bits to carriers
    def setcarrierbitstr(self,blockstr):

        # check out dimensions of blockstr
        blockpercarrier=len(blockstr[0])

        # if we ommit the zeroth subcarrier then assign no bits to it
        if self.ommitzero:
           block2=[''] * blockpercarrier
           blockstr2=[block2]
           blockstr2.extend(blockstr)
        else:
           blockstr2=blockstr

        self.bic=blockstr2

    # read input bit sequence and assign symbol sequences to subcarriers - removes bits from input bit stream
    def setbitstocarriers(self):

        # check if we need to ommit the zeroth subcarrier
        if self.ommitzero:
           nobitspercarrier = np.log2(self.M[1:]).astype(int)
        else:
           nobitspercarrier = np.log2(self.M).astype(int)

        # read the bits
        blocks,bitsremaining,noframes=bitblockscyc2D(self.bin,nobitspercarrier)

        # assign bit blocks to carriers
        self.setcarrierbitstr(blocks)

    def setsymbolmaps(self):
        """
        Set up symbol maps for subcarriers
        """
        self.backwardmaps = []
        self.forwardmaps = []
        self.backwardmapsn = []
        self.forwardmapsn = []
        self.scalesforcarriers = np.zeros( self.nocarriers )
            
        for i in range(0,self.nocarriers):
            fm,bm,fmn,bmn,s = qammap( self.M[i] )
            self.backwardmaps.append( bm )
            self.forwardmaps.append( fm )
            self.backwardmapsn.append( bmn )
            self.forwardmapsn.append( fmn )
            self.scalesforcarriers[i] = s
            
    # assign symbols to carriers by reading the input bits - removes bits from input bit stream
    def setcarriersymbols(self,debug=False):
         
         # assign bits to carriers
         self.setbitstocarriers()

         # create array for symbol storage.
         self.sic = np.zeros([self.nocarriers,self.noframes]) + 1j * np.zeros([self.nocarriers,self.noframes])
         self.sicun = np.zeros([self.nocarriers,self.noframes]) + 1j * np.zeros([self.nocarriers,self.noframes])
         
         for nc in range(0,self.nocarriers):
           blocks=self.bic[nc]
           if debug:
               print('Carrier: %d) has modulation order %d and blocks:' %(nc,self.M[nc]))
               print(blocks)


           for ib,block in enumerate(blocks):

               # Check for subcarrier modulation
              if self.carriermodulation == 'qam':
                  if block != '':
                    q = self.forwardmaps[nc][block]
                    qn = self.forwardmapsn[nc][block]
                  else:
                    q = 0
                    qn = 0
              self.sic[nc,ib] = qn
              self.sicun[nc,ib] = q
           if debug:
                   print('Carrier %d,Block %d bit sequence %s corresponds to symbol %6.2f+j%6.2f' %(nc,ib,block,np.real(q),np.imag(q)))
           if debug:
               print('\n')
    
    # calculate input frames to the ifft block of the transmitter
    def calcifftinput(self):
    
        self.txinputifftframes = []
        for nf in range(0,self.noframes):

            frame = self.sic[:,nf]
            self.txinputifftframes.append( addconjugates( np.sqrt(self.scales) * frame ))

    # calculate out frames of the ifft block at the transmitter
    def calcifftoutput(self):

        self.txoutputifftframes = []

        for frame in self.txinputifftframes:
            ifftout = np.fft.ifft ( frame )
            if self.removeimags:
               self.txoutputifftframes.append ( np.real (ifftout) )
            else:
               self.txoutputifftframes.append ( ifftout )

    def calccptxframes(self):
        """
        add cyclic prefix to frames
        """
        self.txframeswithcp = []
        for i,frame in enumerate(self.txoutputifftframes):
            self.txframeswithcp.append(addcp(frame,self.cpsize))

    def calctxsymbols(self):
        """
        calculate output symbol sequence to be fed to the TX DAC
        """
        self.txs=self.txframeswithcp[0]

        if self.noframes > 0:
            for i in range(1,self.noframes):
                self.txs=np.concatenate((self.txs,self.txframeswithcp[i]))
        
        self.powertx = np.mean( np.abs( self.txs ) ** 2.0 ) # power of the digital signal

    def cliptxsamples(self):
        """
        Clip the samples at the TX output
        """
        
        if not (self.cliplevel is None):
           s = self.powertx                     
           R = 10.0 ** (self.cliplevel/10.0)
           A = np.sqrt(R * s)
           self.Aclip = A
           
           i=np.where( np.abs(self.txs) > self.Aclip)
           self.txsunclipped = np.copy(self.txs)           
           self.txs[i] = self.Aclip * np.sign( self.txs[i] )
           
    def normalizetxs(self):
        """
        Normalize transmitted samples so that they fall inside [-1, 1]
        """
        self.txsu = np.copy(self.txs) 
        self.txs = self.txs / self.Aclip
    
    def applytxnl(self):
        """
        Apply nonlinearity polynomial at the transmitter side
        """
        
        # linear version of the transmitted samples
        self.txsl = np.copy(self.txs)
        
        # apply nonlinearity
        self.txs = polyval(self.txs, self.polynl)
        
    def makedc(self):
        """
        Renders the ouput waveform to DC waveform
        Normalizes the output of the transmitter so that it corresponds to an output level 
        determined by self.Amax
        """
        
        self.txsac = np.copy(self.txs)
        
        if self.Aclip is None:
            self.DC = np.max( np.abs(self.txs) )
            self.Amax = 2.0 * np.max( np.abs(self.txs) )
            self.txs = self.txs + self.DC
        else:
            self.DC = 1.0
            self.Amax = 2.0
            self.txs = self.txs + self.DC
    
    def calctxwaveform(self):
        """
        calculate analog waveform at DAC output
        """
        
        Dt=self.Ts/self.samplespersymbol
        tduration = ( self.framesbefore + self.framesafter + self.noframes ) * self.Tframe

        tinitial = self.framesbefore * self.Tframe

        """
        set analog time and make sure there are even points in the time axis, this is needed when
        calculating the frequency axis
        """
        self.analogt = np.arange(0,tduration,Dt)

        if self.analogt.size % 2 == 1:
            tend=self.analogt[self.analogt.size-1]+Dt
            self.analogt=np.concatenate((self.analogt,np.array([tend])))

        # calculate analog waveform
        self.analogtx = converttoanalog(self.analogt,self.txs,self.Ts,
                                        t0=tinitial,gfilterbandwidth=self.dacfilterbandwidth)

        # check if we need to center time axis
        if self.centertimeaxis:
            Nt=self.analogt.size
            self.analogt = self.analogt - self.analogt[int(Nt/2.0)]

    def calcfreqaxis(self):
        """
        Estimate the frequency axis for the analog output of the TX DAC
        """
        
        self.freqaxis = calcfreqaxis( self.analogt )

    def setchanneltrf(self):
        """
        define the analog channel transfer function
        fun can be a callable or a numpy array
        if the frequency axis does not exist, it is created
        """
            
        # is the frequency axis created?
        if self.freqaxis==None:
            self.calcfreqaxis()

        #  check if the user has supplied a function or a numpy array
        self.trf = self.trfcallable(self.freqaxis)
        
    def applyanalogchannel(self):
        """
        calculation of the input at the receiver input taking into account
        the transfer function
        """

        _,inputspec =spec( self.analogt, self.analogtx )
        self.analogtxspec = inputspec
        outputspec = inputspec * self.trf
        self.analogrxspec = outputspec
        _,output = invspec( self.freqaxis,outputspec )
        self.analogrx = np.real(output)
        
        if not (self.psd is None):
            n = noise( self.analogt, psd = self.psd, )
            self.analogrx = self.analogrx + n
            
    def calcadcoutput(self):
        """
        calculation of the ADC digital output at the receiver
        """
        tmin = np.min(self.analogt)
        tinitial = self.framesbefore * self.Tframe + tmin
        tduration = self.noframes * self.Tframe
        
        self.ts,self.rxs = sample(self.analogt,self.analogrx,self.Ts,
                                  toffset=self.toffset,tinitial=tinitial,
                                  tduration=tduration)
        self.rxsd=self.rxs

    def calcanalogimpulseresponse(self):
        """
        calculate analog channel impulse response
        """
        _,self.ht = invspec( self.freqaxis, self.trf )

    def calctaps(self):
        """
        calculate digital channel taps
        """

        # single pulse p(t) produced by the DAC but use a different time axis than before
        tmin = -self.tapsguardperiod*self.Ts
        tmax = self.tapsguardperiod*self.Ts + self.tapsize*self.Ts
        Dt = self.Ts/self.tapssamplesperTs
        t = np.arange(tmin,tmax,Dt)

        if t.size % 2 == 1:
            tend=t[t.size-1]+Dt
            t=np.append(t,tend)

        # single DAC slice
        p = converttoanalog(t,np.array([1]),self.Ts,
                            t0=0.0,gfilterbandwidth=self.dacfilterbandwidth)

        # calculate the slice spectrum
        f,P = spec(t,p)

        # estimate frequency response of the channel
        if callable(self.trfcallable):
           H = self.trfcallable(f)

        # calculate the output spectrum
        Q = P * H

        # calculate the output waveform
        _,q = invspec(f,Q)

        self.taptimes,self.taps = sample(t,q,self.Ts,toffset=self.toffset,
                                         tinitial=0.0,tduration=self.tapsize*self.Ts)

        self.taps = np.real(self.taps)

    def applydigitalchannel(self):
        """
        calculate the received samples at output of receiver ADC assuming
        only the digital channel
        """
        rxs = np.convolve(self.taps,self.txs)
        # discard unnecessary symbols
        self.rxs0 = np.real(rxs[0:self.txs.size])
        
        if not (self.psd is None):
            t = np.arange(0 , self.rxs0.size) * self.Ts            
            self.n = noise( t, psd = self.psd )
            self.rxs = self.rxs0 + self.n
    
    def removeDC(self):
        """
        Remove DC component from receiver samples
        """
        self.rxs = self.rxs - np.mean(self.rxs)
        
    def normalizerxs(self):
        """
        normalize so that signal power is the same at the receiver and the transmitter        
        """        
        self.rxs = self.Aclip * self.rxs
        
    def calcrxframes(self):
        """
        Separate incoming RX samples to frames
        """
        self.rxframeswithcp = []
        start = 0
        end = self.framesamples

        if self.noframes > 0:
            for i in range(0,self.noframes):
                frame = self.rxs[start:end]
                self.rxframeswithcp.append(frame)
                start=end
                end=end+self.framesamples
                
    def removecprxframes(self):
        """
        Remove CP from RX frames
        """
        start = self.cpsize

        self.rxinputfftframes=[]
        for frame in self.rxframeswithcp:
            framewithoutcp = frame[start:]
            self.rxinputfftframes.append(framewithoutcp)

    def calcfftoutput(self):
        """
        Calculate the output to the FFT module of the receiver and perform equalization
        """
        self.rxoutputfftframes = []

        # if no equalization taps are provided just use the inverse of the channel's tap FFT
         
        if not( isinstance(self.eqtaps, (np.ndarray, np.generic) ) ):
           self.eqtaps = np.zeros(2*self.nocarriers)
           self.eqtaps[0:self.taps.size] = self.taps
           self.eqtaps = 1 / np.fft.fft(self.eqtaps)
        
        
        for frame in self.rxinputfftframes:
            fftout = np.fft.fft ( frame )
            self.rxoutputfftframes.append ( fftout * self.eqtaps )

    def calcrxcarriersamples(self):
        """
        Calculate RX samples at subcarrier channels
        """

        self.rxsic = np.zeros( [self.nocarriers, self.noframes] ) + 1j * np.zeros( [self.nocarriers, self.noframes] )
        self.rxsicun = np.zeros( [self.nocarriers, self.noframes] ) + 1j * np.zeros( [self.nocarriers, self.noframes] )
        
        for i,frame in enumerate(self.rxoutputfftframes):
            self.rxsic[:,i] = frame[0:self.nocarriers] / np.sqrt( self.scales )
            self.rxsicun[:,i] = self.rxsic[:,i] * self.scalesforcarriers
            
    def calcrxestimates(self):
        """
        Calculate symbol and bit estimates at the carriers
        """
        self.siest = np.zeros([self.nocarriers,self.noframes])      \
                   +1j * np.zeros([self.nocarriers,self.noframes])
        self.rxbic = []
        
        for nc in range(0,self.nocarriers):
        
            # Symbol constellation for the carrier at hand
            # bi, co = symconstellation(order = self.M[nc])
            
            carrierbits = []
            for nf in range(0,self.noframes):
                
                if (nc==0) and (self.ommitzero):
                   self.siest[nc,nf] = 0.0
                   carrierbits.append('')
                   
                elif self.carriermodulation == 'qam':
                   
                   sic = self.rxsicun[nc,nf]
                   [siest, biest] = findclosestanddecode( sic, self.backwardmaps[nc] )
                   self.siest[nc,nf] = siest
                   carrierbits.append(biest)
            
            self.rxbic.append(carrierbits)       
                   
    def calcrxbits(self):
        """
        Calculates bits at the output of the receiver
        """
        self.bout=''
        for nf in range(0,self.noframes):
            for nc in range(0,self.nocarriers):           
                self.bout = self.bout + self.rxbic[nc][nf]
                
    def calccrestfactor(self):
        """
        Calculate the crest factor at the transmitter side
        """
        # make sure we are talking about AC-coupled and not DC-coupled signals
        
        txsac = self.txs - np.mean(self.txs)
        
        xmax = np.max( np.abs(txsac) )
        xrms = rms( txsac )
        self.crestfactor = 20*np.log10( xmax / xrms )
        
    def calcerrors(self):
        """
        Calculate bit errors
        """
        self.berrorsinc = np.zeros( self.nocarriers )
        for i in range(0,self.nocarriers):
            b1 = bitstrblockstobitstr(self.rxbic[i])
            b2 = bitstrblockstobitstr(self.bic[i])            
            
            self.berrorsinc[i] = counterrors(b1,b2)
        
        self.berrors = np.sum(self.berrorsinc)
    
    def calcber(self):
        """
        Calculate bit erro rate
        """
        self.calcerrors()
        self.BER = self.berrors / len(self.bout)
                        
    def calcsnrevm(self):
        """
        Calculate receiver SNR at the various channels
        """
        self.snr = np.zeros(self.nocarriers)
        self.snrb = np.zeros(self.nocarriers)
        self.evmp = np.zeros(self.nocarriers)
        
        for i in range(0,self.nocarriers):
          s = np.mean( np.abs( self.sicun[i,:] )**2.0 )
          n = np.mean( np.abs( self.rxsicun[i,:]-self.sicun[i,:] )**2.0 )
          
          if (n != 0) and (s!=0):
             self.snr[i] = 10.0*np.log10( s/n )             
             self.evmp[i] = np.sqrt( n/s ) * 100.0
             self.snrb[i] = self.snr[i] / np.log2(self.M[i])
          else:
             self.snr[i] = np.inf
             self.evmp[i] = np.inf
             self.snrb[i] = np.inf
        
    def digsimulate(self):
        """
        Perform digital channel simulation
        """
        self.timings.total = 0.0
        self.timings.exectimes = {}
        
        for func in self.seqdig:
            toexecute = 'self.' + func +'()'           
            t1 = time.time()
            exec(toexecute)
            t2 = time.time()
            self.timings.exectimes[func] = t2-t1
            self.timings.total += t2-t1
            
    def anlsimulate(self):
        """
        Perform digital channel simulation
        """
        self.timings.total = 0.0
        self.timings.exectimes = {}
        for func in self.seqanl:
            toexecute = 'self.' + func +'()'           
            t1 = time.time()
            exec(toexecute)
            t2 = time.time()
            self.timings.exectimes[func] = t2-t1
            self.timings.total += t2-t1
        
    def showtimings(self):
        """
        Print timing record
        """
        
        di = self.timings.exectimes
        print('')        
        print('Execution Times')
        print('')
        
        for x in di:
            toprint = '%20s : %10.2f ms %6.2f %%'
            print( toprint %(x,di[x]*1000,di[x]/self.timings.total * 100) )
        print('')
        print('Total Execution Time : %10.2f ms' % (self.timings.total * 1000) )
            
    def visualize(self,**kwargs):
        """
        Visualization of various staff at the transmitter or receiver side
        """

        if 'what' in kwargs:

            what=kwargs['what']

            # Channel taps
            if what == 'taps':
               fig=plt.figure()
               plt.plot(np.real(self.taps),'o')
               plt.title('Channel Taps')
               plt.xlabel('tap index')
               plt.ylabel('Tap')

            # samples
            if what == 'samples':

                frameno = kwargs['frameno'] if ('frameno' in kwargs) else list(range(0,self.noframes))
                frameno = [frameno] if isinstance(frameno,int) else frameno
                where = kwargs['where'] if ('where' in kwargs) else 'TXRX'


                start=1
                end=start+self.framesamples

                if 'TX' in where:
                    fig=plt.figure()
                    plt.title('TX samples')

                    for i in frameno:
                        indxs = np.arange(start,end,1,dtype=int)
                        plt.plot(indxs,np.real(self.txframeswithcp[i]),'-o')
                        start=end
                        end=end+self.framesamples

                if 'RX' in where:
                    fig=plt.figure()
                    plt.title('RX samples')

                    for i in frameno:
                        indxs = np.arange(start,end,1,dtype=int)
                        plt.plot(indxs,np.real(self.rxframeswithcp[i]),'-o')
                        start=end
                        end=end+self.framesamples

                if ('TXRX' in where) or ('RXTX' in where):

                    fig=plt.figure()
                    plt.title('TX/RX samples')
                    for i in frameno:
                        indxs = np.arange(start,end,1,dtype=int)
                        plt.plot(indxs,np.real(self.rxframeswithcp[i]),'-o',label='RX')
                        plt.plot(indxs,np.real(self.txframeswithcp[i]),'x',label='TX')
                        start=end
                        end=end+self.framesamples

            # Analog waveforms
            if what == 'analog':
                fig=plt.figure()
                plt.title('Analog waveforms')
                plt.plot(self.analogt/self.Ts,self.analogtx,label='analog TX')
                plt.plot(self.analogt/self.Ts,self.analogrx,label='analog RX')
                plt.plot(self.ts/self.Ts,self.rxs,'o',label='RX samples (a)')
                plt.plot(self.ts/self.Ts,self.rxsd,'o',label='RX samples (d)')
                plt.legend()
                plt.xlabel('$t/T_s$')
                plt.ylabel('Samples')

            # Analog spectra
            if what == 'spectra':
                fig=plt.figure()
                plt.title('Analog spectra')
                plt.plot(self.freqaxis/self.Df,np.abs(self.analogtxspec),label='TX spectrum')
                plt.plot(self.freqaxis/self.Df,np.abs(self.analogrxspec),label='RX spectrum')
                plt.legend()
                plt.xlabel('$fT_s$')
                plt.ylabel('Spectrum')
                
            # samples in carriers            
            if what == 'samples in carriers':

                carrierno = kwargs['carrierno'] if ('carrierno' in kwargs) else list(range(0,self.nocarriers))
                carrierno = [carrierno] if isinstance(carrierno,int) else carrierno

                for i in carrierno:
                   fig, axs = plt.subplots(2)
                   fig.suptitle('Sample comparison carrier '+str(i))
                   axs[0].plot(np.real(self.rxsic[i,:]),'o',label='RX')
                   axs[0].plot(np.real(self.sic[i,:]),'x',label='TX')
                   axs[0].set(ylabel='Real',xlabel='sample index')
                   axs[0].legend()
                   axs[1].plot(np.imag(self.rxsic[i,:]),'o',label='RX')
                   axs[1].plot(np.imag(self.sic[i,:]),'x',label='TX')
                   axs[1].set(ylabel='Imag',xlabel='sample index')
                   axs[1].legend()

            # unscaled samples
            if what == 'unscaled samples':

                carrierno = kwargs['carrierno'] if ('carrierno' in kwargs) else list(range(0,self.nocarriers))
                carrierno = [carrierno] if isinstance(carrierno,int) else carrierno

                for i in carrierno:
                   fig, axs = plt.subplots(2)
                   fig.suptitle('Sample comparison carrier '+str(i))
                   axs[0].plot(np.real(self.rxsicun[i,:]),'o',label='RX')
                   axs[0].plot(np.real(self.sicun[i,:]),'x',label='TX')
                   axs[0].set(ylabel='Real',xlabel='sample index')
                   axs[0].legend()
                   axs[1].plot(np.imag(self.rxsicun[i,:]),'o',label='RX')
                   axs[1].plot(np.imag(self.sicun[i,:]),'x',label='TX')
                   axs[1].set(ylabel='Imag',xlabel='sample index')
                   axs[1].legend()

            # symbols
            if what == 'symbols':

                carrierno = kwargs['carrierno'] if ('carrierno' in kwargs) else list(range(0,self.nocarriers))
                carrierno = [carrierno] if isinstance(carrierno,int) else carrierno

                for i in carrierno:
                   fig, axs = plt.subplots(2)
                   fig.suptitle('Symbol comparison carrier '+str(i))
                   axs[0].plot(np.real(self.siest[i,:]),'o',label='RX')
                   axs[0].plot(np.real(self.sic[i,:]),'x',label='TX')
                   axs[0].set(ylabel='Real',xlabel='symbol index')
                   axs[0].legend()
                   axs[1].plot(np.imag(self.siest[i,:]),'o',label='RX')
                   axs[1].plot(np.imag(self.sic[i,:]),'x',label='TX')
                   axs[1].set(ylabel='Imag',xlabel='symbol index')
                   axs[1].legend()
            
            # Constellation diagrams       
            if what == 'constellations':
            
                carrierno = kwargs['carrierno'] if ('carrierno' in kwargs) else list(range(0,self.nocarriers))
                carrierno = [carrierno] if isinstance(carrierno,int) else carrierno

                for i in carrierno:
                   fig, axs = plt.subplots(1,2)
                   fig.suptitle('Constellation at carrier '+str(i))
                   axs[0].plot(np.real(self.sic[i,:]),np.imag(self.sic[i,:]),'o')
                   axs[0].set(ylabel='Imag',xlabel='Real')
                   axs[0].axis('equal')
                   axs[1].plot(np.real(self.rxsic[i,:]),np.imag(self.rxsic[i,:]),'o')
                   axs[1].set(ylabel='Imag',xlabel='Real')
                   axs[1].axis('equal')
                   
            # plot bits       
            if what == 'bits':
                bitsin = bitstrtobits(self.bin)
                bitsout = bitstrtobits(self.bout)
                
                fig, axs = plt.subplots(2)
                fig.suptitle('Bits at TX and RX')
                axs[0].plot(bitsin,'o',label='TX')
                axs[0].set(ylabel='bit',xlabel='bit index')
                axs[0].legend()

                axs[1].plot(bitsout,'o',label='RX')
                axs[1].set(ylabel='bit',xlabel='bit index')
                axs[1].legend()
                
                diff = np.abs(bitsin-bitsout)
                fig = plt.figure()
                plt.plot( diff )
                plt.xlabel('bit index')
                plt.ylabel('Errors')
                plt.title('Errors')
                
            # plot errors
            if what == 'errors':
                fig = plt.figure()
                plt.plot(self.berrorsinc,'-o')
                plt.xlabel('Carrier no')
                plt.ylabel('Number of errors')
                plt.title('Total errors : %d' %self.berrors)
                
            # plot snr
            if what == 'snr':
                fig = plt.figure()
                plt.plot(self.snr,'-o')
                plt.xlabel('Carrier no')
                plt.ylabel('SNR [dB]')
                plt.title('Receiver SNR')
                
                fig = plt.figure()
                plt.plot(self.snrb,'-o')
                plt.xlabel('Carrier no')
                plt.ylabel('SNRb [dB]')
                plt.title('Receiver SNRb')
                
                fig = plt.figure()
                plt.plot(self.evmp,'-o')
                plt.xlabel('Carrier no')
                plt.ylabel('EVM [%]')
                plt.title('Error Vector Magnitude')
            
            # constellation mapping
            if what == 'symbol mapping':
                carrierno = kwargs['carrierno'] if ('carrierno' in kwargs) else list(range(0,self.nocarriers))
                carrierno = [carrierno] if isinstance(carrierno,int) else carrierno

                for i in carrierno:                    
                    bits = [x for x in self.forwardmaps[i]]
                    symbols = [ self.forwardmaps[i][x] for x in self.forwardmaps[i] ]
                    fig = plt.figure()
                    x = np.real(symbols)
                    y = np.imag(symbols)
                    plt.scatter(x,y,c='r')
                    for i,xx in enumerate(x):
                        plt.text(x[i],y[i],bits[i])
                    plt.title('Carrier no: '+str(i))
                    plt.xlabel('Real')
                    plt.ylabel('Imag')
    
                
"""
Example code for execution of simulation
"""

#Always initialize with the same seed
np.random.seed(0)
random.seed(0)

# Close all figures

plt.close('all')

# Initialization
nocarriers = 16
Fmax = 10e6
order = 16


# Transmitter specs
phy=dmtphy(noframes = 1000,
           samplespersymbol = 10,
           tapsize = 10,
           nocarriers = nocarriers,
           Df = Fmax/nocarriers
           )

#Channel specs


f0 = 10e6
t0 = 1.0e-6
trf = lambda x: 1.0/(1j*x/f0+1)
ht = lambda t: expres(2.0*np.pi*f0,t)
Hf = lambda f: 5e-6 #1e-1* 1/(1+f**2/f0**2)

"""
#Setup transmitter
"""
phy.cliplevel = 20.0
phy.psd = Hf
phy.trfcallable = trf


phy.digsimulate()
phy.showtimings()

phy.visualize(what = 'snr' )


phy.visualize(what = 'errors' )
phy.visualize(what = 'constellations', carrierno=[1,15])
phy.visualize(what = 'symbol mapping',carrierno=1)
phy.visualize(what = 'constellations',carrierno=1)