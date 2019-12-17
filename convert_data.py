import numpy as np

#%%
def convertMonths(months, length):
    x = np.zeros((length,4))
    
    for i in range(length):
        if months[i] in [12,1,2]:
            x[i,0] = 1 # winter
            
        elif months[i] in [3,4,5]:
            x[i,1] = 1 # spring
            
        elif months[i] in [6,7,8]:
            x[i,2] = 1 # summer
            
        else:
            x[i,3] = 1 # autumn
            
    return x

#%%
def convertHours(hours, length):
    x = np.zeros((length, 5))
    
    for i in range(length):
        if hours[i] in list(range(6,10)):
            x[i,0] = 1 # early morning
            
        elif hours[i] in list(range(10,15)):
            x[i,1] = 1 # midday
            
        elif hours[i] in list(range(15,19)):
            x[i,2] = 1 # afternoon
            
        elif hours[i] in list(range(19,23)):
            x[i,3] = 1 # evening
            
        else:
            x[i,4] = 1 # night
            
    return x

#%%
def convertCBWD(cbwd, length):
    x = np.zeros((length, 4))
    
    for i in range(length):
        if cbwd[i] == 'cv':
            x[i,0] = 1
            
        elif cbwd[i] == 'NW':
            x[i,1] = 1
            
        elif cbwd[i] =='NE':
            x[i,2] = 1
            
        else:
            x[i,3] = 1 # SE
    return x