# ignore certain galaxies in the COS samples that don't have counterparts in Simba

import numpy as np

ignore_gals_dict_wrong_imf = {'m50n512_halos': {'ignore_cos_gals':np.array([18, 29]), 
                                        'ngals_each' : 5},
                    'm25n512_dwarfs': {'ignore_cos_gals':np.array([10, 17, 36]), 
                                        'ngals_each':5},
                    'm25n512_halos': {'ignore_cos_gals':np.array([1,  3, 10, 14, 15, 17, 18, 20, 23, 24, 26, 30, 33, 34, 35, 36, 37, 38, 40, 41, 42]), 
                                        'ngals_each' : 3}, 
                    'm25n256_dwarfs': {'ignore_cos_gals':np.array([3,  4,  5,  8, 14, 19, 31, 32, 33, 35, 36, 37]), 
                                        'ngals_each': 4}, 
                    'm25n256_halos': {'ignore_cos_gals':np.array([0,  1,  2,  5, 10, 13, 14, 15, 17, 18, 24, 26, 29, 30, 31, 32, 33, 34, 37, 39, 40, 41, 42]), 
                                        'ngals_each': 4}}

ignore_gals_dict_second_wrong_imf = {'m100n1024_halos': {'ignore_cos_gals':np.array([33, 40, 18]),
                                        'ngals_each' : 5},
                    'm100n1024_dwarfs': {'ignore_cos_gals':np.array([37]),
                                        'ngals_each' : 5},
                    'm50n512_halos': {'ignore_cos_gals':np.array([33, 40, 42, 18, 23]),
                                        'ngals_each' : 4},
                    'm50n512_dwarfs': {'ignore_cos_gals':np.array([37]),
                                        'ngals_each' : 4},
                    'm25n512_halos': {'ignore_cos_gals':np.array([0, 2, 3, 4, 5, 6, 10, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42]),
                                        'ngals_each':3},
                    'm25n512_dwarfs': {'ignore_cos_gals':np.array([8, 14, 15, 20, 32, 33, 34, 35, 36, 37, 38]),
                                        'ngals_each':3},
                    'm25n256_halos': {'ignore_cos_gals':np.array([2, 4, 5, 8, 10, 13, 14, 15, 17, 18, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]),
                                        'ngals_each':3},
                    'm25n256_dwarfs': {'ignore_cos_gals':np.array([4, 14, 15, 19, 24, 30, 32, 33, 34, 35, 36, 37, 38]),
                                        'ngals_each':3}}

ignore_gals_dict = {'m50n512_dwarfs': {'ignore_cos_gals':np.array([35]),
                                        'ngals_each' : 4},
                    'm25n512_dwarfs': {'ignore_cos_gals':np.array([31, 35]),
                                        'ngals_each':4},
                    'm25n512_halos': {'ignore_cos_gals':np.array([1, 3, 5, 10, 14, 17, 18, 21, 23, 24, 29, 31, 32, 33, 34, 35, 37, 38, 40, 41, 42, 43]),
                                        'ngals_each' :4},
                    'm25n256_dwarfs': {'ignore_cos_gals':np.array([1, 3, 4, 5, 8, 13, 14, 19, 20, 22, 28, 30, 31, 32, 33, 34, 35, 36]),
                                        'ngals_each': 4},
                    'm25n256_halos': {'ignore_cos_gals':np.array([0, 1, 5, 7, 10, 14, 15, 18, 21, 23, 24, 25, 29, 30, 32, 33, 34, 35, 37, 40, 41, 42, 43]),
                                        'ngals_each': 4}}

def get_ignore_cos_gals(model, survey):
    if model+'_'+survey in list(ignore_gals_dict.keys()):
        ignore_cos_gals = ignore_gals_dict[model+'_'+survey]['ignore_cos_gals']
        ngals_each = ignore_gals_dict[model+'_'+survey]['ngals_each']
    else:
        ignore_cos_gals = []
        ngals_each = 5
    return ignore_cos_gals, ngals_each

def make_ignore_mask(n, ignore):
    mask = np.ones(n, dtype='bool')
    mask[ignore] = False
    return mask

def get_ignore_cos_mask(model, survey):
    if survey == 'halos':
        ngals = 44
    elif survey == 'dwarfs':
        ngals = 37
    ignore_gals, _ = get_ignore_cos_gals(model, survey)
    return make_ignore_mask(ngals, ignore_gals)

def get_ignore_simba_gals(model, survey):
    ignore_cos_gals, ngals_each = get_ignore_cos_gals(model, survey)
    ignore_simba_gals = [list(range(num*ngals_each, (num+1)*ngals_each)) for num in ignore_cos_gals]
    ignore_simba_gals = [item for sublist in ignore_simba_gals for item in sublist]
    return ignore_simba_gals, ngals_each

def get_ignore_los(ignore_simba_gals):
    nlos = 8
    ignore_los = [list(range(num*nlos, (num+1)*nlos)) for num in ignore_simba_gals]
    ignore_los = np.array([item for sublist in ignore_los for item in sublist])
    return ignore_los
