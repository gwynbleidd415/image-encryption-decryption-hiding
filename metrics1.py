import numpy as np
import pandas as pd
import cv2
from collections import Counter
import matplotlib.pyplot as plt

#Original Image



class Metrics():
    
    def npcr(self,c1, c2):
        M, N = c1.shape
        return (np.sum(c1!=c2))*100/(M*N)

    def uaci(self,c1,c2):
        M, N = c1.shape
        diff = (c1-c2)/255
        s = np.sum(diff)*100
        return s/(M*N)

    # print("'''''''''''''''NPCR ANALYSIS''''''''''''''")
    # print( "B component " , npcr(orig[:,:,0],enc[:,:,0]))
    # print( "G component " , npcr(orig[:,:,1],enc[:,:,1]))
    # print( "R component " , npcr(orig[:,:,2],enc[:,:,2]))

    # print("'''''''UACI Analysis''''''''''''''''''")
    # print( "B component " , uaci(orig[:,:,0],enc[:,:,0]))
    # print( "G component " , uaci(orig[:,:,1],enc[:,:,1]))
    # print( "R component " , uaci(orig[:,:,2],enc[:,:,2]))

    def correlation_coefficient(self,img ):
        M, N = img.shape[0], img.shape[1]
        directions = [ [0,1],[1,0],[1,1]]
        ccfx = []
        for d in directions:
            pix = []
            adj = []
            for i in range(1000):
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
    
    def information_entropy(self,channel):
        M, N = channel.shape[0], channel.shape[1]
        fl = channel.flatten()
        bits = [0]*8
        for no in fl:
            binstr = format(no,'08b')
            for i in range(8):
                if binstr[i]=='1':
                    bits[i]+=1
        ch = 1
        if(len(channel.shape)==3):
            ch = channel.shape[2]
        
        # print(np.array(bits)/(M*N*ch))
        ent= 0
        for i in range(8):
            pi = bits[i]/(M*N*ch)
            ent = ent - pi*np.log2(pi)

        # ent = -ent
        return ent

    # print("'''''''''''''''''''''''''CORRELATION ANALYSIS''''''''''''''''''''''")

    # print("***ORIGINAL IMAGE")
    # cfx_orig = correlation_coefficient(orig[:,:,0])
    # print("B component  (Horiz , Vertical , Diagonal) => " , cfx_orig[0], cfx_orig[1], cfx_orig[2] )
    # cfx_orig = correlation_coefficient(orig[:,:,1])
    # print("G component  (Horiz , Vertical , Diagonal) => " , cfx_orig[0], cfx_orig[1], cfx_orig[2] )
    # cfx_orig = correlation_coefficient(orig[:,:,2])
    # print("R component  (Horiz , Vertical , Diagonal) => " , cfx_orig[0], cfx_orig[1], cfx_orig[2] )


    # print("****ENCRYPTED IMAGE****")
    # cfx_enc = correlation_coefficient(enc[:,:,0])
    # print("B component  (Horiz , Vertical , Diagonal) => " , cfx_enc[0], cfx_enc[1], cfx_enc[2] )
    # cfx_enc = correlation_coefficient(enc[:,:,1])
    # print("G component  (Horiz , Vertical , Diagonal) => " , cfx_enc[0], cfx_enc[1], cfx_enc[2] )
    # cfx_enc = correlation_coefficient(enc[:,:,2])
    # print("R component  (Horiz , Vertical , Diagonal) => " , cfx_enc[0], cfx_enc[1], cfx_enc[2] )


    # print("-------------INFORMATION ENTROPY--------------")
    # print( "B component " , information_entropy(enc[:,:,0]))
    # print( "G component " , information_entropy(enc[:,:,1]))
    # print( "R component " , information_entropy(enc[:,:,2]))







def plot_histogram(img, title="", xlabel="", ylabel="", clr="red"):
    img_fl = img.flatten()
    count = Counter(img_fl)
    plt.bar(count.keys(),count.values(),color=clr)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# plt.figure(figsize= (13,15))
# plt.subplot('231')
# plot_histogram(orig[:,:,2].flatten(),"Lenna Image Red Channel", "","Frequencies",'red')
# plt.subplot('232')
# plot_histogram(orig[:,:,0].flatten(),"Lenna Image Blue Channel", "","Frequencies",'blue')
# plt.subplot('233')
# plot_histogram(orig[:,:,1].flatten(),"Lenna Image Green Channel", "","Frequencies",'green')
# plt.subplot('234')
# plot_histogram(enc[:,:,2].flatten(),"Encrypted Image Red Channel", "Graylevel","Frequencies",'red')
# plt.subplot('235')
# plot_histogram(enc[:,:,0].flatten(),"Encrypted Image Blue Channel", "Graylevel","Frequencies",'blue')
# plt.subplot('236')
# plot_histogram(enc[:,:,1].flatten(),"Encrypted Image Green Channel", "Graylevel","Frequencies",'green')
# plt.show()

names = ['lenna.png','Cameraman.jpg','baboon.jpg','Pepper.png']


info_entro  = []
correlation = []
NPCI = []

for name in names:
    orig =  cv2.imread('../imgs/' + name)
    enc = cv2.imread('../Block Division Encryption Algorithm/encrypted/' + name[:-4]+'.png')
    obj = Metrics()

    cors = []
    # blue = obj.npcr(enc[:,:,0],orig[:,:,0]);
    # red = obj.npcr(enc[:,:,2],orig[:,:,2]);
    # green = obj.npcr(enc[:,:,1],orig[:,:,1]);

    # npcr_uaci = []
    # npcr_uaci.append(np.min([blue,green, red]))
    # npcr_uaci.append(np.max([blue,green, red]))
    # npcr_uaci.append(np.average([blue,green, red]))


    # blue = obj.uaci(enc[:,:,0],orig[:,:,0]);
    # red = obj.uaci(enc[:,:,1],orig[:,:,1]);
    # green = obj.uaci(enc[:,:,2],orig[:,:,2]);

    # npcr_uaci.append(np.min([blue,green, red]))
    # npcr_uaci.append(np.max([blue,green, red]))
    # npcr_uaci.append(np.average([blue,green, red]))

    # NPCI.append(npcr_uaci)
    # info_entro.append(obj.information_entropy(enc))

    blue = obj.correlation_coefficient(enc[:,:,0])
    green  = obj.correlation_coefficient(enc[:,:,1])
    red = obj.correlation_coefficient(enc[:,:,2])

    blue_orig = obj.correlation_coefficient(orig[:,:,0])
    green_orig  = obj.correlation_coefficient(orig[:,:,1])
    red_orig = obj.correlation_coefficient(orig[:,:,2])


    cors.append(np.average([blue_orig[0],red_orig[0],green_orig[0]]))
    cors.append(np.average([blue[0],red[0],green[0]]))
    cors.append(np.average([blue_orig[1],red_orig[1],green_orig[1]]))
    cors.append(np.average([blue[1],red[1],green[1]]))
    cors.append(np.average([blue_orig[2],red_orig[2],green_orig[2]]))
    cors.append(np.average([blue[2],red[2],green[2]]))
    correlation.append(cors)


# columns = ["Min", "Max","Average"]*2
# NPCI = np.array(NPCI)

# df = pd.DataFrame(NPCI, index=[x[:-4] for x in names] , columns=columns,)
# df.to_csv('NPCR_UACI.csv')

# info_entro= np.reshape(np.array(info_entro), (1,len(names)))
# df_  = pd.DataFrame(info_entro, index=["Information Entropy"], columns=[x[:-4] for x in names] )

# df_.to_csv('info_entro.csv')


# cor = np.array(correlation)
# cor = pd.DataFrame(cor)
# cor.to_csv('cor.csv')