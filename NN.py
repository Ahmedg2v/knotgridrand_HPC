import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import sys

def wrapped_sta(sta,size):
    writhe=[]
    for k in range(len(sta)):
        kn = sta[k].reshape(sta[k].shape[0],1)
        extra=size-len(kn)
        #after=random.choice(range(extra))
        #before=extra-after
        #print(before,len(kn),after,np.sum((before,len(kn),after)))
        writhe.append(np.pad(kn,((0,extra),(0,0)),mode='wrap'))
        
    return np.array(writhe)
    
def minmax_norm(x,a,b):
    x_new = a + (x-np.min(x,axis=1, keepdims=True))*(b-a)/(np.max(x,axis=1, keepdims=True)-np.min(x,axis=1, keepdims=True))
    
    return x_new
    
def tanh_norm(x):
    x_new = np.tanh(x/57)       # 57 is max abs writhe value from datasets (std=8 so no extra squeezing)
    return(x_new)
    
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        pass
    
    def backward(self, output_gradient):
        pass
    
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size,1)
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        
    def forward(self, input):
        self.input = input.reshape(input.shape[0], -1)
        return self.input@self.weights.T + self.bias.T

    def backward(self, grad):
        self.dweights = grad.T@self.input
        self.dbias = np.sum(grad, axis=0, keepdims=True).T
        grad = grad@self.weights
        return grad

class Activation(Layer):
    def __init__(self, activation, activation_drv):
        self.activation = activation
        self.activation_drv = activation_drv
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, grad):
        grad = grad*self.activation_drv(self.input)
        return grad

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        
        def relu_drv(x):
            return (x > 0)*1

        super().__init__(relu, relu_drv)
        
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_drv(x):
            return 1-(np.tanh(x)**2)
        
        super().__init__(tanh, tanh_drv)
        
class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = tmp/np.sum(tmp,axis=1,keepdims=True)
        return self.output
    
class Cross_Entropy:
    def forward(output,real):
        entropy= -np.sum(real*np.log(np.clip(output, 1e-12, 1.0)))/len(output)
        return entropy
    
    def backward(output,real):
        grad=(output - real) / len(output)
        return grad
      
class SGD:
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate
        
    def update(self, layer):
        if isinstance(layer,Dense):
            layer.weights -= layer.dweights*self.learning_rate
            layer.bias -= layer.dbias*self.learning_rate
            #eps=1e-8
            #norms = np.linalg.norm(layer.weights, axis=1, keepdims=True)
            #layer.weights /= (norms + eps)
            
class Adam:
    def __init__(self,learning_rate,beta1,beta2,eps):
        self.learning_rate=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.t=0
        
    def update(self,layer):
        if isinstance(layer,Dense):
            layer.m_w = self.beta1*layer.m_w + (1-self.beta1)*layer.dweights
            layer.v_w = self.beta2*layer.v_w + (1-self.beta2)*(layer.dweights**2)
            m_w_hat = layer.m_w/(1-self.beta1**self.t)
            v_w_hat = layer.v_w/(1-self.beta2**self.t)
            
            layer.weights -= self.learning_rate*m_w_hat/(np.sqrt(v_w_hat)+self.eps)
            #norms = np.linalg.norm(layer.weights, axis=1, keepdims=True)
            #layer.weights /= (norms + self.eps)
            
            layer.m_b = self.beta1*layer.m_b + (1-self.beta1)*layer.dbias
            layer.v_b = self.beta2*layer.v_b + (1-self.beta2)*(layer.dbias**2)
            m_b_hat = layer.m_b/(1-self.beta1**self.t)
            v_b_hat = layer.v_b/(1-self.beta2**self.t)
            
            layer.bias -= self.learning_rate*m_b_hat/(np.sqrt(v_b_hat)+self.eps)

            
def add_pts(arr, target):
    #centre = np.array([np.average(arr[:,0]),np.average(arr[:,1]),np.average(arr[:,2])])
    #arr=arr-centre
    mids=(np.vstack((arr[1:],arr[0]))+arr)/2

    mask=np.full((len(mids),1),np.nan)
    mask[0:target-len(arr)]=1
    p=np.random.permutation(len(mask))
    mask=mask[p]
    mids=mids*mask

    tmp=np.zeros((len(mids)+len(arr),arr.shape[-1]))
    tmp[::2]=tmp[::2]+arr
    tmp[1::2]=tmp[1::2]+mids

    new_arr = tmp[np.isfinite(tmp).all(axis=1)]
    
    if (len(new_arr)<target):
        new_arr = add_pts(new_arr, target)
    else:
        new_arr = new_arr 
    
    #new_arr = np.vstack((new_arr[1:],new_arr[0]))-new_arr
    return new_arr
    
def rotate(pts,rng):
    quat = R.random(random_state=rng).as_quat()
    rot = R.from_quat(quat)
    centre = np.array([np.average(pts[:,0]),np.average(pts[:,1]),np.average(pts[:,2])])
    return rot.apply(pts-centre)
    
def batch_gen_3(arr0,arr1,arr2,batch_size,target_size,rng):
    data=[]
    labels=[]
    
    for m in range(batch_size):
        ch=random.choice([0,1,2])

        if ch==0:
            idx = random.choice(range(len(arr0)))
            #data.append(add_pts(arr0[idx],target_size).flatten())
            data.append(add_pts(rotate(arr0[idx],rng),target_size).flatten())
            #data.append(arr0[idx].flatten())
            labels.append([1,0,0])
        
        if ch==1:
            idx = random.choice(range(len(arr1)))
            #data.append(add_pts(arr1[idx],target_size).flatten())
            data.append(add_pts(rotate(arr1[idx],rng),target_size).flatten())
            #data.append(arr1[idx].flatten())
            labels.append([0,1,0])
        
        if ch==2:
            idx = random.choice(range(len(arr2)))
            #data.append(add_pts(arr2[idx],target_size).flatten())
            data.append(add_pts(rotate(arr2[idx],rng),target_size).flatten())
            #data.append(arr2[idx].flatten())
            labels.append([0,0,1])
    
    data=np.array(data, dtype=np.float64)
    labels=np.array(labels, dtype=np.float64)
    
    return data,labels
            
def batch_gen_2(arr0,arr1,batch_size,target_size,rng):
    data=[]
    labels=[]
    
    for m in range(batch_size):
        ch=random.choice([0,1])

        if ch==0:
            idx = random.choice(range(len(arr0)))
            data.append(add_pts(rotate(arr0[idx],rng),target_size).flatten())  # add pts and rotate xyz
            #data.append(add_pts(arr0[idx], target_size).flatten())            # add pts xyz
            #data.append(arr0[idx].flatten())                                   # writhe 2200
            labels.append([1,0])
        
        if ch==1:
            idx = random.choice(range(len(arr1)))
            data.append(add_pts(rotate(arr1[idx],rng), target_size).flatten())
            #data.append(add_pts(arr1[idx], target_size).flatten())
            #data.append(arr1[idx].flatten())
            labels.append([0,1])
        
    #data=wrapped_sta(data, pad_size)
    #temp=[]
    #for k in range(len(data)):
     #   temp.append(add_pts(data[k],target_size).flatten())
    #p=rng.permutation(len(data))
    #data=data[p]
    #labels=labels[p]
    
    data=np.array(data, dtype=np.float64)
    labels=np.array(labels, dtype=np.float64)

    return data,labels

def Train_2(type_1,type_2, model_2, task_id, batch_size, learning_rate, epochs):
    if type_1==0:
        arr_1=arr_0['xyz']
        #arr_1=wr_0['sta']
    elif type_1==3:
        arr_1=arr_3_1['xyz']
        #arr_1=wr_3_1['sta']
    else:
        arr_1=arr_4_1['xyz']
        #arr_1=wr_4_1['sta']
        
    if type_2==0:
        arr_2=arr_0['xyz']
        #arr_2=wr_0['sta']
    elif type_2==3:
        arr_2=arr_3_1['xyz']
        #arr_2=wr_3_1['sta']
    else:
        arr_2=arr_4_1['xyz']
        #arr_2=wr_4_1['sta']

    base_seed = 123456 + task_id*1000
    model = model_2
    optimizer = SGD(learning_rate)
    
    errors=[]
    norms=[]
    accuracy=[]
    for k in range(epochs):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        
        batch=batch_gen_2(arr_1,arr_2 ,batch_size,2200,rng)
        #output=tanh_norm(batch[0])
        output=batch[0]
        real=batch[1]
    
        for layer in model:
            output = layer.forward(output)
    
        accuracy.append(np.sum(np.argmax(output, axis=1) ==  np.argmax(real, axis=1)) / len(np.argmax(output, axis=1)))
    
        errors.append(Cross_Entropy.forward(output,real))
        grad=Cross_Entropy.backward(output,real)
    
        for layer in reversed(model[:-1]):
            grad = layer.backward(grad)
        
        #optimizer.t+=1
        for layer in model:
            optimizer.update(layer)
    
        #norms.append(np.linalg.norm(model[4].dweights))
    
    errors=np.array(errors)
    norms=np.array(norms)
    accuracy=np.array(accuracy)
    
    w1, b1 = model[0].weights, model[0].bias
    w2, b2 = model[2].weights, model[2].bias
    #w3, b3 = model[4].weights, model[4].bias
    w3, b3 = [],[]
    
    np.savez_compressed(f'model_parameters_1024rot_{type_1},{type_2}_xyz_{batch_size}bsize005SGD{epochs}e', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, errors=errors, norms=norms, accuracy=accuracy)
    
def Train_3(type_1,type_2,type_3, model_3, task_id, batch_size, learning_rate, epochs):
    arr_1=arr_0['xyz']
    arr_2=arr_3_1['xyz']
    arr_3=arr_4_1['xyz']
    
    #arr_1=wr_0['sta']
    #arr_2=wr_3_1['sta']
    #arr_3=wr_4_1['sta']

    base_seed = 123456 + task_id*1000
    model = model_3
    optimizer = SGD(learning_rate)
    
    errors=[]
    norms=[]
    accuracy=[]
    for k in range(epochs):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        
        batch=batch_gen_3(arr_1,arr_2,arr_3 ,batch_size,2200,rng)
        #output=tanh_norm(batch[0])
        output=batch[0]
        real=batch[1]
    
        for layer in model:
            output = layer.forward(output)
    
        accuracy.append(np.sum(np.argmax(output, axis=1) ==  np.argmax(real, axis=1)) / len(np.argmax(output, axis=1)))
    
        errors.append(Cross_Entropy.forward(output,real))
        grad=Cross_Entropy.backward(output,real)
    
        for layer in reversed(model[:-1]):
            grad = layer.backward(grad)
        
        #optimizer.t+=1
        for layer in model:
            optimizer.update(layer)
    
        #norms.append(np.linalg.norm(model[4].dweights))
    
    errors=np.array(errors)
    norms=np.array(norms)
    accuracy=np.array(accuracy)
    
    w1, b1 = model[0].weights, model[0].bias
    w2, b2 = model[2].weights, model[2].bias
    #w3, b3 = model[4].weights, model[4].bias
    w3, b3 = [],[]
    
    np.savez_compressed(f'model_parameters_1024rot_{type_1},{type_2},{type_3}_xyz_{batch_size}bsize005SGD{epochs}e', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, errors=errors, norms=norms, accuracy=accuracy)

#wr_3_1=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__3_1_wr_100_2200.npz',allow_pickle=True)
#wr_4_1=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__4_1_wr_100_2200.npz',allow_pickle=True)
#wr_0=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__unknot_wr_100_2200.npz',allow_pickle=True)

arr_0=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__unknot_xyz.npz',allow_pickle=True)
arr_3_1=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__3_1_xyz.npz',allow_pickle=True)
arr_4_1=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__4_1_xyz.npz',allow_pickle=True)

task_id = int(sys.argv[1])

#rng = np.random.default_rng(seed)

#p=rng.permutation(len(data))
#data=data[p]
#labels=labels[p]
#optimizer = Adam(0.001,0.9,0.999,1e-8)
#optimizer = SGD(0.01)

batch_size=128
epochs=50
learning_rate=0.005

model_2=[Dense(6600,1024),
         ReLU(),
         Dense(1024,2),
         Softmax()]

model_3=[Dense(6600,1024),
         ReLU(),
         Dense(1024,3),
         Softmax()]
         
#       Dense(128,32),
#       ReLU(),       
#         Dense(256,256),
#         ReLU(),


def main(task_id):
    if task_id==0:
        Train_2(0,3, model_2, task_id, batch_size, learning_rate, epochs)
    
    if task_id==1:
        Train_2(0,4, model_2, task_id, batch_size, learning_rate, epochs)
        
    if task_id==2:
        Train_2(3,4, model_2, task_id, batch_size, learning_rate, epochs)
        
    if task_id==3:
        Train_3(0,3,4, model_3, task_id, batch_size, learning_rate, epochs)

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
