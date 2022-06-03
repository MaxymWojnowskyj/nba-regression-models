import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm

        

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

# obtains the base distribution stats for an array of numbers
def getDistStats(x): 
    mean = sum(x)/len(x) # mean of x value (average x value)
    var = sum((x - mean)**2) # variance of x (expectation of how much values deviate from the mean)
    std = np.sqrt(var) # average value for how much a value deviates from the mean
    return {"Mean": mean, "Var": var, "Std": std}

mins_stats = getDistStats(mins)
pts_stats = getDistStats(pts)
 

# covariance, the larger the number the stronger the relationship between the variables. pos = pos relationship, neg = neg relationship
covar = sum ( (mins - mins_stats["Mean"]) * (pts - pts_stats["Mean"]) )

#corr = covar/(mins_stats["Std"]*pts_stats["Std"])

w1 = covar/mins_stats["Var"]
w0 = pts_stats["Mean"] - w1*mins_stats["Mean"]
#w0 = 0
#w1 = 0.008

print(f"y = {w0} + {w1}x")


tr_mins_stats = getDistStats(tr_mins)
tr_pts_stats = getDistStats(tr_pts)

# covariance, the larger the number the stronger the relationship between the variables. pos = pos relationship, neg = neg relationship
tr_covar = sum( (tr_mins - tr_mins_stats["Mean"]) * (tr_pts - tr_pts_stats["Mean"]) )

tr_w1 = tr_covar/tr_mins_stats["Var"]
tr_w0 = tr_pts_stats["Mean"] - tr_w1*tr_mins_stats["Mean"]

print(f"y = {tr_w0} + {tr_w1}x")


#Gradient descent, create cost func and plot cost vs w0 and w1 valuess
# predict function given time spent playing (in seconds) output predicted pts scored
def predict(w0, w1, x): 
    return w0 + w1*x # returns array given array of x's

# objective function output sum of squared error
def objective(w0, w1, x, y): # this should return an array instead of a single value
    # adding 2 to (2*len(x)) as the 2 makes the derivation simpler and the 2 only affects the actual error/residual value but does not change at which weight value it is minimized (x values stay the same but at changed y values)
    return  sum((predict(w0, w1, x) - y)**2) / (2*len(x))

def deriv_of_obj(w1, x, y): # derivative of predict w/ respect to w1
    return  sum((predict(0, w1, x) - y)*x) / (len(x)) 


# starts at zero as a player cant score less than zero points
w1_trials = np.arange(0, 2*tr_w1+0.0001, 0.0001) # covariance optimized predictor cant be less than half the actual optimization so make that the max

w1_obj = [objective(0, w1_trial, tr_mins, tr_pts) for w1_trial in w1_trials]

# randomly selecting a starting point for grad desc
rand_w1 = np.random.choice(w1_trials)

fig1, ax1 = plt.subplots()

ax1.set_title('Grad Desc on SLR model')
ax1.set_xlabel('Pts scored per second played (W1)')
ax1.set_ylabel('Sum of Squared Residuals (SSR)')

ax1.plot(w1_trials, w1_obj, '-', color='orange', label='objective')
# plot the starting random gradient point plot( x val rand_pt = weight value (w), y value of rand_pt = objective of that weight value, shape, colour of shape)
ax1.plot([rand_w1], [objective(0, rand_w1, tr_mins, tr_pts)], 's', color='g', label='Random start')

# gradient descent with backtracking line search general purpose selection for step size

beta = 0.7 # reduce step size by 30% until no longer over steps
new_w1 = rand_w1
step_count = 0

# while the derivative of our current weight is not approximatly equal to zero (while new_w1 has not hit a stationary point)
while not np.isclose(abs(deriv_of_obj(new_w1, tr_mins, tr_pts)), 0, atol=0.001):
    
    obj_deriv_new_w1 = deriv_of_obj(new_w1, tr_mins, tr_pts) 
    
    # reset alpha (step size) for each iteration as we recalculate it
    alpha = 1
    # formula taken from: https://youtu.be/4qDt4QUl4zE
    while objective(0, new_w1 - alpha*obj_deriv_new_w1, tr_mins, tr_pts) > objective(0, new_w1, tr_mins, tr_pts) - (alpha/2)*(obj_deriv_new_w1**2):
        alpha = alpha*beta   

    step_count += 1
    new_w1 = new_w1 - alpha * obj_deriv_new_w1 

print("Steps taken:", step_count)
print("new FINAL wt", new_w1)

#plt.plot(w1_trials, w1_obj, '-', label='objective')
# plot the starting random gradient point plot( x val rand_pt = weight value (w), y value of rand_pt = objective of that weight value, shape, colour of shape)
ax1.plot([new_w1], [objective(0, new_w1, tr_mins, tr_pts)], 's', color='r', label='Minima ('+"{:.4f})".format(new_w1)) # only display new_w1 up to 4 decimal places
ax1.legend()
# create residual vs fits plot to see if linear model is appropriate
# y-axis: residuals of each point
# x-axis: each point is the players points scored

fig2, ax2 = plt.subplots()
ax2.set_title('Residual vs Fit')
ax2.set_xlabel('Pts')
ax2.set_ylabel('Residual')
residuals = predict(0, new_w1, tr_mins) - tr_pts # an array of the residual for each prediction pts vs actual pts scored given time played
ax2.scatter(tr_pts, residuals)


# barplot to see if residuals are skewed for mins played predictor
fig4, ax4 = plt.subplots()
ax4.bar(tr_pts, abs(residuals))

x = tr_mins
y = tr_pts
order = np.argsort(x)

fig3, ax3 = plt.subplots()


t = [int(s.split("-")[0]) for s in tr_seasons_arr]
ax3.scatter(x, y, c=t, cmap=cm.cool, alpha=0.7)
# can replace w0 with zero
ax3.plot(x, 0 + w1*tr_mins, label='Original')
#can replace tr_w0 with zero
ax3.plot(x, 0 + tr_w1*tr_mins, label='Trimmed')
# squared plot gotten from multivar_ex.py file
ax3.plot(x[order], 0 + 0.0046881011291261335*x[order] + 1.630208666892446e-06*(x[order]**2), color='mediumblue', label='squared')
formatter = matplotlib.ticker.FuncFormatter(lambda sec, x: time.strftime('%M:%S', time.gmtime(sec)))
ax3.xaxis.set_major_formatter(formatter)
ax3.set_title('SLR model for predicted points scored given time played')
ax3.set_xlabel('Time played (Min:Sec)')
ax3.set_ylabel('Points scored')
sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=plt.Normalize(vmin=min(t), vmax=max(t)))#, norm=plt.normalize(min=0, max=1))
fig3.colorbar(sm)
ax3.legend()
plt.show()


