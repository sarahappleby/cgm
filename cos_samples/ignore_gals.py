# ignore certain galaxies in the COS samples that don't have counterparts in Simba

import numpy as np

ignore_gals_dict = {'m50n512_halos': {'ignore_cos_gals':np.array([18, 29]), 
                                        'ngals_each' : 5},
                    'm25n512_dwarfs': {'ignore_cos_gals':np.array([10, 17, 36]), 
                                        'ngals_each':5},
                    'm25n512_halos': {'ignore_cos_gals':np.array([1,  3, 10, 14, 15, 17, 18, 20, 23, 24, 26, 30, 33, 34, 35, 36, 37, 38, 40, 41, 42]), 
                                        'ngals_each' : 3}, 
                    'm25n256_dwarfs': {'ignore_cos_gals':np.array([3,  4,  5,  8, 14, 19, 31, 32, 33, 35, 36, 37]), 
                                        'ngals_each': 4}, 
                    'm25n256_halos': {'ignore_cos_gals':np.array([0,  1,  2,  5, 10, 13, 14, 15, 17, 18, 24, 26, 29, 30, 31, 32, 33, 34, 37, 39, 40, 41, 42]), 
                                        'ngals_each': 4}}

def get_ignore_cos_gals(model, survey):
    ignore_cos_gals = ignore_gals_dict[model+'_'+survey]['ignore_cos_gals']
    ngals_each = ignore_gals_dict[model+'_'+survey]['ngals_each']
    return ignore_cos_gals, ngals_each

def get_ignore_simba_gals(model, survey):
    ignore_cos_gals, ngals_each = get_ignore_cos_gals(model, survey)
    ignore_simba_gals = [list(range(num*ngals_each, (num+1)*ngals_each)) for num in ignore_cos_gals]
    ignore_simba_gals = [item for sublist in ignore_simba_gals for item in sublist]
    return ignore_simba_gals, ngals_each
