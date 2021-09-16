
import random
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

def generate_data(xmin,xmax, delta, noise):
    #Calculate f=sin(x1)+cos(x2)
    x1=np.arange(xmin,xmax+delta,delta) #generate x1 values
    x2=np.arange(xmin, xmax+delta, delta) #generate x2 values
    x1,x2=np.meshgrid(x1,x2) #make x1, x2 grid of values
    f=np.sin(x1)+np.cos(x2) #calculate for all (x1,x2) grid

    #Add random noise to f
    random.seed(123) #set random seed for reproducibility 
    
    for i in range(len(f)):
        for j in range(len(f[0])):
            f[i][j]=f[i][j]+random.uniform(-noise,noise)
    return x1,x2,f

#Transform X into a 2D numpy array and y into a 1D numpy array 
def prepare_data(x1,x2,f):
    X=[]
    for i in range(len(f)):
      for j in range(len(f)):
        X_temp=[]
        X_temp.append(x1[i][j])
        X_temp.append(x2[i][j])
        X.append(X_temp)
    y=f.flatten()
    X=np.array(X)
    y=np.array(y)
    return X,y

  
def KRR_function(hyperparams,X,y, f_provi):
    #Assign hyperparams
    alpha_value,gamma_value=hyperparams
    #Split data into test and train_set: random state fixed for reproducibility 
    kf=KFold(n_splits=10, shuffle=True,random_state=123)
    y_pred=[]
    y_test=[]
    #KFold cross-validation loop

    for train_index, test_index in kf.split(X):
      X_train, X_test=X[train_index], X[test_index]
      y_train, y_test_temp=y[train_index], y[test_index]

      #Scale X_train and X_test
      scaler=preprocessing.StandardScaler().fit(X_train)
      X_train_scaled=scaler.transform(X_train)
      X_test_scaled=scaler.transform(X_test)

      # Fit KRR with scaled datasets 
      KRR=KernelRidge(kernel='rbf',alpha=alpha_value, gamma=gamma_value)
      y_pred_temp=KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)

      #Append y_pred_temp and y_test_temp of this k-fold step to the list 
      y_pred.append(y_pred_temp)
      y_test.append(y_test_temp)

    #Flatten lists with test and predicted values
    y_pred=[i for sublist in y_pred for i in sublist]
    y_test=[i for sublist in y_test for i in sublist]
    
    #Estimate error metric of test and predicted value
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    print('alpha: %.6f . gamma: %.6f . rmse: %.6f' %(alpha_value,gamma_value,rmse)) # Uncomment to print intermediate results
    f_provi.write("%.20f %.20f %.12f\n" %(alpha_value,gamma_value,rmse))
    return rmse