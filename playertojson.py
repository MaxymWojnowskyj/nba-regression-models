import json
import math
import matplotlib
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
#import numpy as np

def percentile(data, perc: int):
    size = len(data)
    return sorted(data)[int(math.ceil((size * perc) / 100)) - 1]

# obtains the base distribution stats for an array of numbers
def getDistStats(x): 
    mean = sum(x)/len(x) # mean of x value (average x value)
    var = 0
    for xi in x: var += (xi - mean)**2 # variance of x (expectation of how much values deviate from the mean)
    std = math.sqrt(var) # average value for how much a value deviates from the mean
    return {"Mean": mean, "Var": var, "Std": std}

def player_prompt():
    print('\n')
    valid_file = False
    user_input = ''
    while not valid_file and user_input != 'q':
        user_input = input("Please input the file name of a player (or press q to quit): ")
        if user_input != 'q':
            try:
                file = open(user_input)
                valid_file = True
            except IOError:
                print("Invalid file name, file not found.")

    if user_input == 'q': return [None]*2

    #otherwise user did not quit so return json and the players name
    return [json.load(file), user_input.split(".json")[0]]

def display_ptsVmin(player, data):
    x = data['mins']
    y = data['pts']
    s = data['seasons']
    fig, ax = plt.subplots()
    #t = [int(s.split("-")[0]) for s in seasons]
    ax.scatter(x, y, c=s, cmap=cm.cool)
    formatter = matplotlib.ticker.FuncFormatter(lambda sec, x: time.strftime('%M:%S', time.gmtime(sec)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title('Points scored vs. time played')
    ax.set_xlabel('Time played (Min:Sec)')
    ax.set_ylabel('Points scored')
    sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=plt.Normalize(vmin=min(s), vmax=max(s)))#, norm=plt.normalize(min=0, max=1))
    fig.colorbar(sm)
    plt.savefig(f'{player}.jpg')

def display_ptsVvar(player, data, var):
    x = data[var]
    y = data['pts']
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(f'Points scored vs. {var}')
    ax.set_xlabel(f'{var}')
    ax.set_ylabel('Points scored')
    plt.savefig(f'{player}_PtsV{var}.jpg')

def outliers(player, data):
    x = data['mins']
    y = data['pts']
    seasons = data['seasons']
    pts_stats = getDistStats(y)
    remove_prompt = input("Enter in y if you would like to remove an outlier and any other key to not remove any outliers: ")
    if remove_prompt == 'y':
        lower, upper = [0, 0]
        if pts_stats['Std'] > pts_stats['Mean']:
            print(f"The standard devition of {player}'s points scored is greater than the mean thus we will remove the outliers via the IQR technique")
            q25, q75 = percentile(y, 25), percentile(y, 75)
            iqr = q75 - q25
            # calculate the outlier cutoff
            cut_off = iqr * 1.5
            lower, upper = q25 - cut_off, q75 + cut_off
    
        else:
            print(f"{player}'s points scored is approximatly normal thus we will remove the outliers via the empirical rule (removing all points outside of 2 standard deviations from the mean 95%)")
            lower, upper = pts_stats['Mean'] - 2*pts_stats["Std"], pts_stats['Mean'] + 2*pts_stats["Std"] 

        outliers = {i:y[i] for i in range(len(x)) if y[i] < lower or y[i] > upper}
        print(f"Outliers with point values: {outliers.values()} removed.")

        # remove outliers from data set (pts and mins array)
        for idx in outliers.keys(): # tr stands for trimmed
            del y[idx]
            del x[idx]
            del seasons[idx]

        print("Saving new trimmed Pts vs. Time played plot:")
        display_ptsVmin(f'{player}_trimmed', data)#{'mins':x, 'pts': y, 'seasons':seasons})

    else: print("No outliers removed")
    print('\n')

def extract_vars(player, data):
    mins = data['mins']
    y = data['pts'] # the variable we are trying to predict
    seasons = data['seasons']
    #seasons = [int(s.split("-")[0]) for s in data['seasons']]
    X = [] # an array containing a variable that we will use to try and predict y (pts) (each variable is a vector)
    available_vars = [key for key in data.keys() if key != 'pts']

    valid_input = False
    user_input = ''
    print("The current available variables to select from is the following list:")
    var_idx = 1
    for var in available_vars:
        print(f"{var_idx} - {var}")
        var_idx += 1

    end_download_dists = False
    while not end_download_dists:
        sel_idx = input("To download a jpg of a variables distribution vs pts scored please input the variables index number, or press any other char to move on to downloading the data into json files: ")
        try:
            sel_idx = int(sel_idx)
            sel_idx -= 1
            if sel_idx < 0 or sel_idx > len(available_vars):
                print(f"Selected index is out of range, please select an index between 1 and {len(available_vars)}.")
            else: # otherwise number is in valid range so download jpg of its distribution
                sel_var = available_vars[sel_idx]
                display_ptsVvar(player, data, sel_var) 
                #display_ptsVvar(player, {'mins':mins, 'pts': y, 'seasons':seasons}, sel_var)
                print(f"{player}_PtsV{sel_var}.jpg downloaded\n")

        except ValueError as error: end_download_dists = True

    print('\n')

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    user_input = ''
    valid_input = False
    while not valid_input and user_input != 'q':
        print("Input a list of your desired predictor variables seperated by commas (,). Exponential: x^#, Log: x_log#, Div: x/# (To multiply var divide by 1/# and enter in decimal form 0.#).")
        idx = 0
        for var in available_vars:
            if idx > 26: # if we have more than 26 vars then reuse alph for vars but add number to end (based on how many times we have used alphabet)
                print(f"{var}: {alphabet[(idx%26)-1] + str(math.floor(idx/26)+1)}")
            else: print(f"{var}: {alphabet[idx]}")
            idx += 1 
        # ex input: a, a^2, b, a2, 2b, log2_c 
        # a2 is var at index 27, 2b is the second variable b multiplied/scaled by 2
        user_input = input("Variables list (or enter q to quit): ")
        if user_input != 'q':
            X = []
            user_input = user_input.replace(' ', '') # remove spaces from input
            for var in user_input.split(','):
                # default splits should strip the var of all transformation strings should be left with ex: 'a', 'a2', 'a#...#'
                str_var = var.split('_')[0].split('^')[0].split('/')[0] 
                var_idx = alphabet.index(str_var[0]) # default take index of first char in string (usually (wont be more than 26 vars))
                if len(str_var) > 1: var_idx = alphabet.index(str_var[0]) + (int(str_var[1:])-1)*26 # ex b2 = 1 + (2-1)*26 = 27
                sel_var = available_vars[var_idx]
                if '^' in var: X.append([float(val)**float(var.split('^')[1]) for val in data.get(sel_var)]) # each val in var array exponentially multiplied by var.split('^')[1] 
                elif 'log' in var: X.append([math.log(float(val), int(var.split('log')[1])) for val in data.get(sel_var)]) # each val in var array logged with base var.split('log')[1] 
                elif '/' in var: X.append([float(val)/float(var.split('/')[1]) for val in data.get(sel_var)]) # each val in var array divided by var.split('/')[1]
                else: X.append(data.get(sel_var)) # add the vars array into the X array (no transformation applied)
            valid_input = True
            X_filename = input("Input the name for the X variables list json file (.json will already be included): ")
            y_filename = input("Input the file name for the pts list json file or enter n to not download the pts json file: ")
            with open(X_filename+'.json', 'w') as f: json.dump(X, f)
            print(f"{X_filename}.json saved")
            if y_filename != 'n': 
                with open(y_filename+'.json', 'w') as f: json.dump(y, f) 
                print(f"{y_filename}.json saved")



def main():
    data, player = player_prompt()
    if data == None: return
    #otherwise proceed with conversion
    mins, pts, seasons_arr = [], [], []

    for season in data:
        for game in data[season]['Games']:
            str_time = game['Player']['Min']
            sec_time = int(str_time.split(":")[0])*60 + int(str_time.split(":")[1]) # mins*60 + remaining seconds = total seconds
            mins.append(sec_time)
            pts.append(int(game['Player']['Pts']))
            seasons_arr.append(int(season.split('-')[0])) 

    data = {"mins": mins, "pts": pts, "seasons": seasons_arr}

    print("Saving players pts vs time played jpg...\n")

    display_ptsVmin(player, data)
    print(len(data['seasons']))
    outliers(player, data)
    print(len(data['seasons']))
    extract_vars(player, data)



main()