import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./artifacts/bang_home_price_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location




'''
get_estimated_price----------------
This is a function in Python that calculates an estimated price for a property based on its location, square footage, number of bedrooms, and
 number of bathrooms.

It starts by trying to find the index of the location in a list called __data_columns. If the location is not found, it assigns the value of -1 to 
the loc_index.

Then, it creates a zero array x of length len(__data_columns) and sets the values of x[0], x[1], and x[2] to the square footage, number of
 bathrooms, and number of bedrooms, respectively. If the location is found in __data_columns, the value of x[loc_index] is set to 1.

Finally, the function returns the result of the prediction by the __model for the input array x, rounded to two decimal places.





'''