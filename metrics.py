import numpy as np
import cv2

#Original Image
orig =  cv2.imread('./imgs/lenna_color.png')
orig = cv2.resize(orig, (256,256))

#Encrypted Image
enc = cv2.imread('fuck.png')



def npcr(c1, c2):
    M, N = c1.shape
    return (np.sum(c1!=c2))*100/(M*N)

def uaci(c1,c2):
    M, N = c1.shape
    diff = (c1-c2)/255
    s = np.sum(diff)*100
    return s/(M*N)

print("'''''''''''''''NPCR ANALYSIS''''''''''''''")
print( "B component " , npcr(orig[:,:,0],enc[:,:,0]))
print( "G component " , npcr(orig[:,:,1],enc[:,:,1]))
print( "R component " , npcr(orig[:,:,2],enc[:,:,2]))

print("'''''''UACI Analysis''''''''''''''''''")
print( "B component " , uaci(orig[:,:,0],enc[:,:,0]))
print( "G component " , uaci(orig[:,:,1],enc[:,:,1]))
print( "R component " , uaci(orig[:,:,2],enc[:,:,2]))

def correlation_coefficient( img ):
    M, N = img.shape[0], img.shape[1]
    directions = [ [0,1],[1,0],[1,1]]
    ccfx = []
    for d in directions:
        pix = []
        adj = []
        for i in range(10):
            rx = np.random.randint(0,M-1)
            ry = np.random.randint(0,N-1)
            pix.append(img[rx][ry])
            adj.append(img[rx+d[0]][ry+d[1]])
        pix = np.array(pix)
        adj = np.array(adj)
        a = pix - np.mean(pix)
        b = adj - np.mean(adj)
        num = np.sum(a*b)
        denm = np.sum(a**2) * np.sum(b**2)
        corr = num/np.sqrt(denm)
        corr = np.round(corr,3)
        ccfx.append(corr)
    
    return ccfx

print("'''''''''''''''''''''''''CORRELATION ANALYSIS''''''''''''''''''''''")

print("***ORIGINAL IMAGE")
cfx_orig = correlation_coefficient(orig[:,:,0])
print("B component  (Horiz , Vertical , Diagonal) => " , cfx_orig[0], cfx_orig[1], cfx_orig[2] )
cfx_orig = correlation_coefficient(orig[:,:,1])
print("G component  (Horiz , Vertical , Diagonal) => " , cfx_orig[0], cfx_orig[1], cfx_orig[2] )
cfx_orig = correlation_coefficient(orig[:,:,2])
print("R component  (Horiz , Vertical , Diagonal) => " , cfx_orig[0], cfx_orig[1], cfx_orig[2] )


print("****ENCRYPTED IMAGE****")
cfx_enc = correlation_coefficient(enc[:,:,0])
print("B component  (Horiz , Vertical , Diagonal) => " , cfx_enc[0], cfx_enc[1], cfx_enc[2] )
cfx_enc = correlation_coefficient(enc[:,:,1])
print("G component  (Horiz , Vertical , Diagonal) => " , cfx_enc[0], cfx_enc[1], cfx_enc[2] )
cfx_enc = correlation_coefficient(enc[:,:,2])
print("R component  (Horiz , Vertical , Diagonal) => " , cfx_enc[0], cfx_enc[1], cfx_enc[2] )

print(pwd)