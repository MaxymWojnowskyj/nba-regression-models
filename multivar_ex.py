import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

file = open('Fred_VanVleet.json')
data = json.load(file)


mins, pts, seasons_arr = [], [], []

for season in data:
    for game in data[season]['Games']:
        str_time = game['Player']['Min']
        sec_time = int(str_time.split(":")[0])*60 + int(str_time.split(":")[1]) # mins*60 + remaining seconds = total seconds
        mins.append(sec_time)
        pts.append(int(game['Player']['Pts']))
        seasons_arr.append(season) 

mins, pts, seasons_arr = np.array(mins), np.array(pts), np.array(seasons_arr)

# remove outlier
# if std is larger than mean than our data is not normal and we cant use empirical rule (68, 95, 99.7)
# try IQR
q25, q75 = np.percentile(pts, 25), np.percentile(pts, 75)
iqr = q75 - q25
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
outliers = {i:pts[i] for i in range(len(pts)) if pts[i] < lower or pts[i] > upper}

# remove outliers from data set (pts and mins array)
for idx in outliers.keys(): # tr stands for trimmed
    tr_pts = np.delete(pts, idx)
    tr_mins = np.delete(mins, idx)
    tr_seasons_arr = np.delete(seasons_arr, idx)
    #del mins[idx]
    #del pts[idx]
    #del seasons_arr[idx]

# obtains the base distribution stats for an array of numbers
def getDistStats(x): 
    mean = sum(x)/len(x) # mean of x value (average x value)
    var = sum((x - mean)**2) # variance of x (expectation of how much values deviate from the mean)
    std = np.sqrt(var) # average value for how much a value deviates from the mean
    return {"Mean": mean, "Var": var, "Std": std}


X = np.array([tr_mins, tr_mins**2, tr_mins**3])
y = tr_pts
#upercase var for matrix, lowercase for vector

def predict(weights, X):
    # sum of:((W*X^T)^T) 
    # ex: f = w1*x + w2*x^2 + w3*z
    # array of weights: W = [w1, w2, w3] (wi is a scalar)
    # matrix of variables: (X = [x, x^2, z, etc..] each var inside X vector is a vector .:. making a matrix) X = [[x1, x2, ..., xn], [x1^2, x2^2, ..., xn^2], [z1, z2, ...zn], etc..]
    return sum(np.transpose(weights*np.transpose(X)))



# objective function output sum of squared error
def objective(weights, X, y): # this should return an array instead of a single value
    # adding 2 to (2*len(x)) as the 2 makes the derivation simpler and the 2 only affects the actual error/residual value but does not change at which weight value it is minimized (x values stay the same but at changed y values)
    return  sum((predict(weights, X) - y)**2) / (2*len(X[0])) #length of each vector in X matrix should be the same so take len of first one (doesnt matter which one)

def deriv_obj_wi(weights, X, y, i):
    #taking partial deriv w/ respect to wi will have the predict multiplied by the term that is multiplied with wi (xi in this case could be x^2*z, x, z, etc.. whatevers multiplied by wi in the predict function)
    return sum((predict(weights, X) - y)*X[i]) / (len(X[0]))  



weights = [0, 0, 0]

# can declare convergence to stationary point when gradient only decreases by small number like 1e^-3 for example.
# while the sum of each of our derivatives is not approximatly equal to zero (while new_w1 has not hit a stationary point)

beta = 0.7 # reduce step size by 30% until no longer over steps
step_count = 0
sum_last_derivs = 1

# sum together the absolute values of each derivative value to continue grad desc until the grad for each weight is not significantly different from the derivative of the last step in grad desc.
while not np.isclose( sum([abs(deriv_obj_wi(weights, X, y, i)) for i in range(len(weights))]) - sum_last_derivs, 0, atol=0.001):

    sum_last_derivs = sum([abs(deriv_obj_wi(weights, X, y, i)) for i in range(len(weights))])

    for i in range(len(weights)):

        obj_deriv_new_wi = deriv_obj_wi(weights, X, y, i)

        print(f"obj_deriv_w{i}: {obj_deriv_new_wi}")
    
        # reset alpha (step size) for each iteration as we recalculate it
        alpha = 1

        #backtracing line search for selecting step size
        # multiple each weight in vector weights by: -alpha*obj_deriv_new_wi*activation (only activation_i should be 1, rest should be zero.)
        #need to multiple vector in objective spot so multiply weights vector by [0, 0, 1, 0, 0, ..., 0]
        activation = np.zeros(len(weights))
        activation[i] = 1
        while objective(weights, X, y) < objective(weights - alpha*obj_deriv_new_wi*activation, X, y):
            alpha = alpha*beta

        step_count += 1
        #update the new weight
        weights[i] = weights[i] - alpha * obj_deriv_new_wi
    
print("Steps taken:", step_count)
print(f"new FINAL weights", weights)
print("Final min objective found:", objective(weights, X, y))
