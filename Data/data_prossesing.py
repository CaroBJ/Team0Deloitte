# function to load the data and modify it to have backgrounds

# Imports
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt # etc. etc.
import cv2
from rembg import remove
import os

def car_data_load(Path_to_data, Path_to_landscapes, num_train, num_test):

    # Load Landscapes
    land_dir_list = os.listdir(Path_to_landscapes)
    land_dir_list.sort()

    #land=np.array(land)

    num_black_car=((num_train)//5)*2
    num_orange_car=((num_train)//5)*2
    num_real_car=((num_train)//5)*1

    total_black=834
    total_orange=2001
    total_real=138

    if num_real_car > total_real:
        num_real_car=total_real

    test_num=30

    dir_list = os.listdir(Path_to_data)
    dir_list.sort()
  
    black_space=total_black//num_black_car
    orange_space=total_orange//num_orange_car
    real_space=total_real//num_real_car
  
    
    train_array=[]

    

    for i in range(0,num_black_car*black_space,black_space):
    #for i in range(1):
        car_im=np.load(os.path.join(Path_to_data, dir_list[i]))
        # Removing background from car image
        car_front=remove(car_im[:,:,0:3])
        mask_front=car_front[:,:,3]<=100
        mask_front_rev=car_front[:,:,3]>100
        for l in range(np.shape(land_dir_list)[0]):
        #for l in range(1):
           #temp_land=land[l,:,:,:]
            temp_land=cv2.imread(os.path.join(Path_to_landscapes, land_dir_list[l]))
            temp_land=temp_land[:,:,::-1]
            temp_land[:,:,0]=temp_land[:,:,0]*mask_front+mask_front_rev*car_im[:,:,0]
            temp_land[:,:,1]=temp_land[:,:,1]*mask_front+mask_front_rev*car_im[:,:,1]
            temp_land[:,:,2]=temp_land[:,:,2]*mask_front+mask_front_rev*car_im[:,:,2]
            final_land= np.dstack((temp_land,car_im[:,:,3]))
            
            train_array.append(final_land)

    for i in range(total_black,(num_orange_car*orange_space)+total_black,orange_space):
        #for i in range(1):
        car_im=np.load(os.path.join(Path_to_data, dir_list[i]))
        # Removing background from car image
        car_front=remove(car_im[:,:,0:3])
        mask_front=car_front[:,:,3]<=245
        mask_front_rev=car_front[:,:,3]>245
        for l in range(np.shape(land_dir_list)[0]):
        #for l in range(1):
            #temp_land=land[l,:,:,:]
            temp_land=cv2.imread(os.path.join(Path_to_landscapes, land_dir_list[l]))
            temp_land=temp_land[:,:,::-1]
            temp_land[:,:,0]=temp_land[:,:,0]*mask_front+mask_front_rev*car_im[:,:,0]
            temp_land[:,:,1]=temp_land[:,:,1]*mask_front+mask_front_rev*car_im[:,:,1]
            temp_land[:,:,2]=temp_land[:,:,2]*mask_front+mask_front_rev*car_im[:,:,2]
            final_land= np.dstack((temp_land,car_im[:,:,3]))
            
            train_array.append(final_land)

    
    

    for i in range(total_black+total_orange+test_num,(num_real_car*real_space)+total_black+total_orange+test_num,real_space):
        car_im=np.load(os.path.join(Path_to_data, dir_list[i]))
        train_array.append(car_im)
        print(dir_list[i])

    test_array=[]

    np.save('Prossed_data_train.npy',np.array(train_array))
   

    for i in range(total_black+total_orange,total_black+total_orange+num_test):
            car_im=np.load(os.path.join(Path_to_data, dir_list[i]))
            test_array.append(car_im)
            print(dir_list[i])
     
    

    # Save data in npy file
    np.save('Prossed_data_test.npy',np.array(test_array))
    return np.array(train_array), np.array(test_array)



# Put in path to data
path=''
path_1=''

# Make the data with 1000 images in train and 30 in test
train,test =car_data_load(path_1, path, 1000, 30)
print(np.shape(train))
print(np.shape(test))
