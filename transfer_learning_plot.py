import numpy as np
import matplotlib.pyplot as plt

# doing this manually, not smart but whatever :P

                    # 0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37     38     39
vmunet_ce_transfer = [0.636, 0.636, 0.714, 0.738, 0.768, 0.789, 0.802, 0.811, 0.829, 0.829, 0.829, 0.829, 0.836, 0.836, 0.838, 0.848, 0.848, 0.848, 0.848, 0.850, 0.850, 0.850, 0.853, 0.853, 0.853, 0.857, 0.857, 0.857, 0.857, 0.859, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860]
vmunet_ce_notransfer = [0.636, 0.636, 0.666, 0.710, 0.742, 0.742, 0.742, 0.770, 0.770, 0.770, 0.770, 0.770, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780, 0.780]

swinunet_ce_transfer = [0.466, 0.487, 0.526, 0.555, 0.567, 0.590, 0.599, 0.599, 0.607, 0.618, 0.618, 0.625, 0.634, 0.634, 0.641, 0.646, 0.646, 0.656, 0.656, 0.658, 0.658, 0.658, 0.658, 0.661, 0.661, 0.661, 0.663, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.666, 0.668, 0.669, 0.669, 0.669, 0.669]
swinunet_ce_notransfer = [0.636, 0.636, 0.636, 0.641, 0.658, 0.686, 0.690, 0.699, 0.699, 0.720, 0.720, 0.726, 0.726, 0.727, 0.727, 0.727, 0.729, 0.729, 0.744, 0.744, 0.744, 0.744, 0.744, 0.744, 0.744, 0.744, 0.744, 0.744, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752]

plt.title("VMUNet/SwinUnet Imagnet1k pretraining dice scores")
plt.plot(vmunet_ce_notransfer, 'r')
plt.plot(vmunet_ce_transfer, 'r', linestyle='dashed')
plt.plot(swinunet_ce_notransfer, 'b')
plt.plot(swinunet_ce_transfer, 'b', linestyle='dashed')
plt.legend(['VMUNet default', 'VMUNet pretrained', 'SwinUnet default', 'SwinUnet pretrained'])
plt.xlabel('epoch')
plt.ylabel('2D dice score')
plt.ylim(0.4, 1) 
plt.show()

plt.title("VMUNet Imagnet1k pretraining dice scores")
plt.plot(vmunet_ce_notransfer, 'r')
plt.plot(vmunet_ce_transfer, 'r', linestyle='dashed')
plt.legend(['VMUNet default', 'VMUNet pretrained'])
plt.xlabel('epoch')
plt.ylabel('2D dice score')
plt.ylim(0.4, 1) 

plt.show()

plt.title("SwinUnet Imagnet1k pretraining dice scores")
plt.plot(swinunet_ce_notransfer, 'b')
plt.plot(swinunet_ce_transfer, 'b', linestyle='dashed')
plt.legend(['SwinUnet default', 'SwinUnet pretrained'])
plt.xlabel('epoch')
plt.ylabel('2D dice score')
plt.ylim(0.4, 1) 

plt.show()