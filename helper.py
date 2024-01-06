import matplotlib.pyplot as plt
from torchvision.models.video import r2plus1d_18,R2Plus1D_18_Weights,swin3d_t,Swin3D_T_Weights,mvit_v2_s,MViT_V2_S_Weights

def interpret_pred(output):
    label = {1:'Healthy Lifestyle and Weight Loss', 
         2:'Weight Lifting', 
         3:'Running',
         4:'Yoga',
         5:'Haircare',
         6:'Makeup',
         7:'Skincare',
         8:'Outfit',
         9:'Accommodation',
         10:'Adventure',
         11:'Culture',
         12:'Food and drink'}
    ans = []
    for idx,y in enumerate(output):
        if y == 1.:
            ans.append(label[idx+1])
    return ans

def plot(frame,total_frames,subplot_row,subplot_col,title=None,adjust_top=1.66,fontsize=10,figsize=(2,4)):
    fig = plt.figure(figsize=figsize,dpi=300)
    
    for i in range(total_frames):
        ax = fig.add_subplot(subplot_row, subplot_col, i + 1)        
        ax.imshow(frame[i].permute(1,2,0))
        ax.axis("off")
    fig.suptitle(title,fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(top=adjust_top)

def models(name,data = None):
    if data != None:
        if name == 'mvit':
            weight = MViT_V2_S_Weights.KINETICS400_V1
            model = mvit_v2_s(weights=weight)
            transforms = weight.transforms()
            trans_data = transforms(data)
            return model,trans_data,weight
        elif name == 'swin':
            weight = Swin3D_T_Weights.KINETICS400_V1
            model = swin3d_t(weights=weight)
            transforms = weight.transforms()
            trans_data = transforms(data)
            return model,trans_data,weight
        elif name == 'resnet':
            weight = R2Plus1D_18_Weights.KINETICS400_V1
            model = r2plus1d_18(weights=weight)
            transforms = weight.transforms()
            trans_data = transforms(data)
            return model,trans_data,weight

    elif data == None:
        if name == 'mvit':
            weight = MViT_V2_S_Weights.KINETICS400_V1
            model = mvit_v2_s(weights=weight)
            return model,weight
        elif name == 'swin':
            weight = Swin3D_T_Weights.KINETICS400_V1
            model = swin3d_t(weights=weight)
            return model,weight
        elif name == 'resnet':
            weight = R2Plus1D_18_Weights.KINETICS400_V1
            model = r2plus1d_18(weights=weight)
            return model,weight