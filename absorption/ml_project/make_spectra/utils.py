import h5py

def read_h5_into_dict(h5file):
    data = {}
    with h5py.File(h5file, 'r') as f:
        for k in f.keys():

            if type(f[k]) == h5py._hl.group.Group:
                data[k] = {}

                for gk in f[k].keys():

                    if len(f[k][gk].shape) == 0:
                        data[k][gk] = f[k][gk][()]
                    else:
                        data[k][gk] = f[k][gk][:]

            else:
                if len(f[k].shape) == 0:
                    data[k] = f[k][()]
                else:
                    data[k] = f[k][:]

                for attr_k in f[k].attrs.keys():
                    data[attr_k] = f[k].attrs[attr_k]

    return data

