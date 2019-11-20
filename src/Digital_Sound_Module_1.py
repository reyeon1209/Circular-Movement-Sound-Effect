import wave, array
import numpy as np
import matplotlib.pyplot as plt


# 파일 분리
def save_wav_channel(fn, wav, channel):
    # Read data
    nch   = wav.getnchannels() # getnchannels() : 오디오 채널 수 반환 (모노 : 1, 스테레오 : 2)
    depth = wav.getsampwidth() # getsampwidth() : 샘플 폭을 바이트 단위로 반환
    wav.setpos(0) # setpos(위치) : 파일 포인터를 지정된 위치로 설정
    sdata = wav.readframes(wav.getnframes()) # readframes(n) : 최대 n개의 오디오 프레임을 bytes 객체로 읽고 반환

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.frombuffer(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w') # open(file, mode) : 파일 열기
    outwav.setparams(wav.getparams()) # setparams(tuple) : parameter 설정
    outwav.setnchannels(1) # setnchannles(n) : 채널 수 설정
    outwav.writeframes(ch_data.tostring()) # writeframes(data) : 오디오 프레임 쓰기
    outwav.close() # close() : 파일 닫기

# 파일 결합
def make_stereo(input1, input2, output): # 함수 인자 : 왼쪽, 오른쪽, 합친 결과물

    wav1 = wave.open(input1) # open(file) : 파일 열기
    wav2 = wave.open(input2)
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav1.getparams()
    # getparams() : namedtuple() 반환 (get*()의 반환값과 같음) 

    assert comptype == 'NONE'  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, wav1.readframes(nframes))[::nchannels]
    right_channel = array.array(array_type, wav2.readframes(nframes))[::nchannels]
          
    # 정보를 활용하는 부분
    print("nchannels", nchannels)
    print("sampwidth", sampwidth)
    print("framerate", framerate)
    print("nframes", nframes)
    print("comptype", comptype)
    print("compname", compname)
    print()
    
    perTime = nframes/framerate
    print(perTime,"초 짜리 음")
    
    # frameRate(고정),바꿀채널이름, arrayType(고정)
    right_channel = rightControl(framerate, right_channel, array_type)
    left_channel = leftControl(framerate, left_channel, array_type)

    printChannel(left_channel, 121)
    printChannel(right_channel, 122)

    wav1.close() # close() : 파일 닫기
    wav2.close() # close() : 파일 닫기
    
    stereo = 2 * left_channel
    stereo[0::2] = left_channel
    stereo[1::2] = right_channel

    ofile = wave.open(output, 'w') # open(file, mode) : 파일 열기
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tobytes())
    ofile.close() # close() : 파일 닫기
    
def rightControl (framerate, channel, array_type): # 오른쪽 파일 변환하는 함수
    startFrame = 0 # 파일의 처음부터
    endFrame = framerate*4 # 파일의 끝까지 (참고 : framerate = 16000)

    channelList = channel.tolist() # list로 변경해서 저장 (수정하기 위해)
    
    for i in range(startFrame, endFrame): # 파일의 처음부터 파일의 끝까지 돌면서
        pos = i/framerate # 현재 위치를 시간으로 나타냄 ex) 0초(=0/64000), 1초, 2초, 3초, 4초
        
        # 2초를 기준으로 적용하는 수식이 바뀜
        if pos < 2:# 2초 전이면
            channelList[i] = int(channelList[i]*np.cos((np.pi/4)*pos))
        
        else: # 2초 후이면
            channelList[i] = int(channelList[i]*(np.cos((np.pi/4)*(pos+2))+1))

    channel = array.array(array_type, channelList) # list로 변경하여 저장했던 것을 array로 변경해서 저장
    return channel # 변환된 array 반환

def leftControl (framerate, channel, array_type): # 왼쪽 파일 변환하는 함수
    startFrame = 0 # 파일의 처음부터
    endFrame = framerate*4 # 파일의 끝까지 (참고 : framerate = 16000)

    channelList = channel.tolist() # list로 변경해서 저장 (수정하기 위해)
    
    for i in range(startFrame, endFrame): # 파일의 처음부터 파일의 끝까지 돌면서
        pos = i/framerate # 현재 위치를 시간으로 나타냄 ex) 0초(=0/64000), 1초, 2초, 3초, 4초
        
        # 2초를 기준으로 적용하는 수식이 바뀜
        if pos < 2: # 2초 전이면
            channelList[i] = int(channelList[i]*(np.cos((np.pi/4)*(pos+2))+1))
            
        else: # 2초 후이면
            channelList[i] = int(channelList[i]*(np.cos((np.pi/4)*(pos-4))))

    channel = array.array(array_type, channelList) # list로 변경하여 저장했던 것을 array로 변경해서 저장
    return channel # 변환된 array 반환

def printChannel (channel, subPlotVar):
    channelList = channel.tolist()
    plt.subplot(subPlotVar)
    plt.plot(channelList)


plt.show()
 
    
wav = wave.open('sample.wav')
save_wav_channel('left.wav', wav, 0)
save_wav_channel('right.wav', wav, 1)
make_stereo('left.wav', 'right.wav', 'sample_result.wav')
