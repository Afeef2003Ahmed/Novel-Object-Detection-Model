from Model import Model
from Model import backbone,Neck,Detect
#from Model_small import Model
import torch

model = Model(num_classes=10,backbone=backbone(),channels_list = [256,512,1024])
#neck = Neck()
''' ******** Model Parameters ********'''
def model_params(model):
    state_dict = model.state_dict()

    #print(model.backbone)

    for key, value in state_dict.items():
        print(key, "\t", value.size())
        
''' ******* Model Size ********** '''

def model_size(model):
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return size_all_mb
    
    
''' ******* Number of Model Parameters ******* '''
    
def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)





def model_info(model):
    
    n_param = count_parameters(model)
    n_param_mil = f"{n_param / 1_000_000:.2f}M"
    print(f"Number of Trainable Parameters in Model is : {n_param_mil}\n")
    
    size_all_mb = model_size(model)
    print('model size: {:.3f}MB \n'.format(size_all_mb)) 
    

#model_params(model)
#print("\n")
model_info(model) 

#print(model.backbone)

'''def main():

    input_shape = (1,3, 416, 416)
    input_tensor = torch.randn(input_shape)
    model = backbone()
    neck = Neck()
    output_backbone = model(input_tensor)
    neck_output = neck(output_backbone)
    for output in neck_output:
        
        print(f'Output Shape: {output.shape} \n')
    
if __name__ == '__main__':
    main()
'''    
