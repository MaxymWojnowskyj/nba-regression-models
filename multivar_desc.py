import json
import numpy as np

def valid_int(prompt_msg, err_msg):
    #prompt to get a positive int from user
    valid_int = False
    while not valid_int:
        user_int = input(prompt_msg)
        try:
            user_int = int(user_int)
            if user_int < 1: print("Please enter a positive number")
            else: valid_int = True
        except ValueError: print(err_msg) 
    return user_int 

def valid_numb(prompt_msg, err_msg):
    # prompts the user to enter a positve integer or float
    valid_numb = False
    while not valid_numb:
        numb = input(prompt_msg)
        try:
            numb = float(numb) # if int then will just be #.0
            valid_numb = True
        except ValueError: print(err_msg) 
    return numb 


def valid_file(data_type, len_weights, X):
    valid_file = False
    user_input = ''
    prompt_msg = ''
    if data_type == 'X':
        prompt_msg = "Please input the name of the json file that contains the X data (X should be a matrix (vector containing vectors, Ex: X = [x, x^2, z], x = [1, 2, 3], z = [0.1, 0.2, 0.3] ): "
    else: prompt_msg = "Please input the name of the json file that contains the y data (y should be a vector containing the values you are trying to predict): "

    while not valid_file:
        file_name = input(prompt_msg)
        if user_input != 'q':
            try:
                file = open(file_name)
                data = json.load(file)
                if data_type == 'X':
                    if len(data) != len_weights:
                        print("selected X file does not have the proper amount of rows/variables.")
                    else: valid_file = True
                else:
                    if len(data) != len(X[0]):
                        print(f"The imported json file for your prediction variables y has {len(data)} values. This does not match your selected number of predictions: {len(X[0])}.")
                    else: valid_file = True
            except IOError:
                print("Invalid file name, file not found.")
    if user_input == 'q': data = None 
    return data

def user_prompt():
    weight_count = valid_int("Please enter the amount of weights you would like in the model: ", "Please enter a whole number (int)")
    weights = []
    for i in range(weight_count):
        weight_guess = valid_numb(f"Enter an int or float as a guess for what you think the value for weight {i} (w{i}) should be (this will be the starting point for gradient descent): ", "Please enter in a valid integer or float number for the weight value (examples: -0.05, 3, 3000, 0.77): ")
        weights.append(weight_guess)

    X = ''
    X = valid_file('X', len(weights), X)
    if X == None: return [None] * 4
    y = valid_file('y', len(weights), X)
    if y == None: return [None] * 4

    # prompts user the enter a valid float between 0 and 1 (decimal value)
    valid_beta = False
    beta = ''
    while not valid_beta and beta != 'q':
        beta = input("Please input a number between 0 and 1 (decimal) as your beta value (value to multiply your step size by in backtracing line search, reducing step size by (1-beta)% until a valid step size is found (or enter q to quit): ")
        if beta != 'q':
            try:
                beta = float(beta)
                print(beta)
                if beta < 0 or beta > 1:
                    print("Please enter a number between 0 and 1") 
                else: valid_beta = True
            except: print("Please enter a valid float number between 0 and 1")

    if beta == 'q': return [None] * 4
    return [X, y, beta, weights]


# first letter uppercase in var for matrix, lowercase for vector
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


def grad_desc(beta, weights, X, y):
    step_count = 0
    sum_last_derivs = 1
    print("Enter control + C to stop gradient descent at current point and get the weights that produced the smallest objective up to that point. (minarg(weights): obj(weights, X, y))")
    try:
        # can declare convergence to stationary point when gradient only decreases by small number like 1e^-3 for example.
        # while the sum of each of our derivatives is not approximatly equal to zero (while new_w1 has not hit a stationary point)
        # sum together the absolute values of each derivative value to continue grad desc until the grad for each weight is not significantly different from the derivative of the last step in grad desc.
        while not np.isclose( sum([abs(deriv_obj_wi(weights, X, y, i)) for i in range(len(weights))]) - sum_last_derivs, 0, atol=0.01):

            sum_last_derivs = sum([abs(deriv_obj_wi(weights, X, y, i)) for i in range(len(weights))])

            deriv_str = ''
            for i in range(len(weights)):

                obj_deriv_new_wi = deriv_obj_wi(weights, X, y, i)

                deriv_str += f" obj_deriv_w{i}: {obj_deriv_new_wi}" 
            
                # reset alpha (step size) for each iteration as we recalculate it
                alpha = 1

                # backtracing line search for selecting step size
                # multiply each weight in vector weights by: -alpha*obj_deriv_new_wi*activation (only activation_i should be 1, rest should be zero.)
                # need to multiply vector in objective spot so multiply weights vector by [0, 0, 1, 0, 0, ..., 0]
                activation = np.zeros(len(weights))
                activation[i] = 1
                while objective(weights, X, y) < objective(weights - alpha*obj_deriv_new_wi*activation, X, y):
                    # if the new objective with wi (updated by wi - alpha*deriv_wi) does not have a smaller objective then the objective at the previous step then decrease the step size (alpha) by multiplying it by beta (decreasing step size by (1-beta)%)
                    alpha = alpha*beta

                step_count += 1
                #update the new weight
                weights[i] = weights[i] - alpha * obj_deriv_new_wi
            
            print(deriv_str, end='\r')

    except KeyboardInterrupt: print("Gradient descent stopped early.")
        
    print("Steps taken:", step_count)
    print(f"new FINAL weights", weights)
    print("Final min objective found:", objective(weights, X, y))


def main():
    X, y, beta, weights = user_prompt()
    # if the user did not quit then proceed with desc
    if X != None: grad_desc(beta, weights, X, y)

main()