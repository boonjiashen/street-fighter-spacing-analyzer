import subprocess
import numpy as np

FFMPEG_BIN = 'ffmpeg'
command = [ FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '420x360', # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a import pipe
        '-an', # Tells # FFMPEG # not # to # expect # any # audio
        '-vcodec',
        'mpeg',
        'output_videofile.mp4' ]

pipe = subprocess.Popen( command, stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)

#for _ in range(1):
#for i in range(1):
image_array = np.ones([100, 100], dtype=np.uint8) 
#pipe.stdin.write( image_array.tostring() )
pipe.communicate(input=image_array.tostring())

pipe.terminate()
