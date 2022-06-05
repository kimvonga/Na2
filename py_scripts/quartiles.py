import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-f','--file',dest='file_in',help='Input file',metavar='file_in',default='')
parser.add_option('-c','--column',dest='column',help='column location of data',metavar='column',default='1')
parser.add_option('-s','--start',dest='start',help='choose when to start array',default='0')

(options, args) = parser.parse_args()
file_in = options.file_in
col = int(options.column)
start = int(options.start)

data = np.loadtxt(file_in)[start:,col]
time = np.loadtxt(file_in)[start:,0]

qs = np.percentile(data,[34,50,84])
print('['+'{:8.4f}'.format(min(data))+','+'{:8.4f}'.format(qs[0])+','+'{:8.4f}'.format(qs[1])+','+
        '{:8.4f}'.format(qs[2])+','+'{:8.4f}'.format(max(data))+']')
#print([min(data),qs[0],qs[1],qs[2],max(data)])
print('average:     '+str(np.average(data)))
print('index of max:    '+str(np.argwhere(data == max(data))[0][0]+1))
