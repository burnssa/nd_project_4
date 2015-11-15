#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    import numpy as np
    from numpy import array

    residual_errors = np.subtract(predictions, net_worths)
    variables = tuple(zip(ages, net_worths, residual_errors))
    sorted_errors = sorted(variables, key = lambda x: x[2])

    cleaner_cutoff = int(0.9 * len(sorted_errors) - 1)
    cleaned_data = list(sorted_errors[0:cleaner_cutoff])

    return cleaned_data

