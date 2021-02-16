from astropy.io import fits, ascii
import h5py


civ_data = '/home/sapple/cgm/cos_samples/obs_data/cos_dwarfs/line_table_simple.tex'
table = ascii.read(table_file, format='latex')
civ_gal_names = np.array(table['Galaxy'])

lya_data = fits.open('/home/sapple/cgm/cos_samples/obs_data/cos_dwarfs/COS-Dwarfs_Lya.fits')
data = lya_data[1].data
lya_gal_names = data['GAL'][0]

corresponding = np.zeros(len(civ_gal_names))

split_lya_list = [i.split('_') for i in lya_gal_names]

for i in range(len(civ_gal_names)):

    name = civ_gal_names[i]
    parts = name.split('\\_')
    find = False
    
    for j in range(len(split_lya_list)):
        if parts == split_lya_list[j]:
            find = True
            corresponding[i] = j
            continue
    if find == False:
        absent = i
        print('Not found for galaxy ' + str(i))
        
corresponding = np.array(corresponding, dtype='int')
new_names_lya = [lya_gal_names[i] for i in corresponding]
new_names_lya[absent] = ''
new_names_lya = np.array(new_names_lya)

new_lya_file = '/home/sapple/cgm/cos_samples/obs_data/cos_dwarfs/lya_data_civ_order.h5'
keys = list(lya_data[1].columns.names)
for k in keys:
    stuff = data[k][0]
    if len(stuff) == 42:
        new_stuff = [stuff[i] for i in corresponding]
        if type(new_stuff[0]) == str:
            new_stuff[absent] = ''
            new_stuff = np.string_(new_stuff)
        elif type(new_stuff[0]) == np.int64:
            new_stuff[absent] = 0
        elif type(new_stuff[0]) == np.float64:
            new_stuff[absent] = 0.

        with h5py.File(new_lya_file, 'a') as f:
            f.create_dataset(k, data=np.array(new_stuff))
