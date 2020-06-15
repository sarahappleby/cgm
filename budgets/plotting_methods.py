import numpy as np 

def get_bin_edges(x_min, x_max, dx):
	return np.arange(x_min, x_max+dx, dx)

def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def variance_jk(samples, mean):
        n = len(samples)
        factor = (n-1.)/n
        x = np.nansum((np.subtract(samples, mean))**2, axis=0)
        x *= factor
        return x

def get_cosmic_variance(quantity, pos, boxsize):
	octant_ids = octants_2d(pos, boxsize)
	measure = np.zeros(8)
	for i in range(8):
		i_using = np.concatenate(np.delete(octant_ids, i))
		measure[i] = np.nanmedian(quantity[i_using.astype('int')])
	mean_m = np.nansum(measure) / 8.
	cosmic_std = np.sqrt(variance_jk(measure, mean_m))
	return mean_m, cosmic_std

def octants_2d(pos_array, boxsize):
    pos_x = (pos_array[:, 0] < boxsize*0.5)
    pos_ya = (pos_array[:, 1] < boxsize*0.25)
    pos_yb = (pos_array[:, 1] > boxsize*0.25)& (pos_array[:, 1] < boxsize*0.5)
    pos_yc = (pos_array[:, 1] > boxsize*0.5)& (pos_array[:, 1] < boxsize*0.75)
    pos_yd = (pos_array[:, 1] > boxsize*0.75)
    inds_1 = np.array([]); inds_2 = np.array([]); inds_3 = np.array([]); inds_4 = np.array([])
    inds_5 = np.array([]); inds_6 = np.array([]); inds_7 = np.array([]); inds_8 = np.array([])

    for i in range(len(pos_array)):
        if pos_ya[i] and pos_x[i]:
            inds_1 = np.append(inds_1, i)
        elif pos_ya[i] and not pos_x[i]:
            inds_2 = np.append(inds_2, i)
        elif pos_yb[i] and pos_x[i]:
            inds_3 = np.append(inds_3, i)
        elif pos_yb[i] and not pos_x[i]:
            inds_4 = np.append(inds_4, i)
        elif pos_yc[i] and pos_x[i]:
            inds_5 = np.append(inds_5, i)
        elif pos_yc[i] and not pos_x[i]:
            inds_6 = np.append(inds_6, i)
        elif pos_yd[i] and pos_x[i]:
            inds_7 = np.append(inds_7, i)
        elif pos_yd[i] and not pos_x[i]:
            inds_8 = np.append(inds_8, i)

    return np.array((inds_1, inds_2, inds_3, inds_4, inds_5, inds_6, inds_7, inds_8))


