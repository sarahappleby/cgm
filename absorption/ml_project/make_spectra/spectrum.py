import numpy as np
import h5py
import matplotlib.pyplot as plt
import pygad as pg
from physics import wave_to_vel, vel_to_wave, tau_to_flux
from utils import read_h5_into_dict


class Spectrum(object):


    def __init__(self, spectrum_file, **kwargs):

        self.spectrum_file = spectrum_file
        for key in kwargs:
            setattr(self, key, kwargs[key])
        data = read_h5_into_dict(self.spectrum_file)
        for key in data:
            setattr(self, key, data[key])
        del data


    def extend_to_continuum(self, vel_range, contin_level=1., nbuffer=20):

        self.vel_mask = (self.velocities < self.gal_velocity_pos + vel_range) & (self.velocities > self.gal_velocity_pos - vel_range)
        v_start, v_end = np.where(self.vel_mask)[0][0], np.where(self.vel_mask)[0][-1]

        continuum = False
        i = 0
        while not continuum:
            _flux = self.fluxes[v_start - i:v_start -i +nbuffer]
            if np.abs(np.median(_flux) - contin_level) / contin_level > 0.05:
                i += 1
            else:
                continuum = True

        continuum = False
        j = 0
        while not continuum:
            _flux = self.fluxes[v_end + j - nbuffer: v_end +j]
            if np.abs(np.median(_flux) - contin_level) / contin_level > 0.05:
                j += 1
            else:
                continuum = True

        extended_indices = np.arange(v_start - i - nbuffer, v_end+j+ nbuffer+1, 1)
        extended_indices = np.delete(extended_indices, np.argwhere(extended_indices > len(self.vel_mask) -1))
        self.vel_mask = np.zeros(len(self.vel_mask)).astype(bool)
        self.vel_mask[extended_indices] = True


    def fit_spectrum(self, vel_range=600., nbuffer=20):

        contin_level = self.continuum[0]
        self.extend_to_continuum(vel_range, contin_level)

        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths[self.vel_mask], self.fluxes[self.vel_mask], self.noise[self.vel_mask],
                                                  chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
    
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)


    def fit_periodic_spectrum(self):

        wrap_flux, wrap_noise, wrap_start = pg.analysis.periodic_wrap(self.wavelengths, self.fluxes, self.noise)
        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths, wrap_flux, wrap_noise,
                                         chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
        self.line_list['l'] = pg.analysis.periodic_unwrap_wavelength(self.line_list['l'], self.wavelengths, wrap_start)
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)


    def get_tau_model(self):
        self.tau_model = np.zeros(len(self.wavelengths))
        for i in range(len(self.line_list["N"])):
            p = np.array([self.line_list["N"][i], self.line_list["b"][i], self.line_list["l"][i]])
            self.tau_model += pg.analysis.model_tau(self.ion_name, p, self.wavelengths)

    
    def get_fluxes_model(self):
        self.get_tau_model()
        self.fluxes_model = np.exp(-np.clip(self.tau_model, -30, 30))

    
    def write_line_list(self):

        with h5py.File(self.spectrum_file, 'a') as hf:
            if 'line_list' in hf.keys():
                del hf['line_list']
            elif 'lines' in hf.keys():
                del hf['lines']
            line_list = hf.create_group("line_list")
            line_list.create_dataset("region", data=np.array(self.line_list['region']))
            line_list.create_dataset("N", data=np.array(self.line_list['N']))
            line_list.create_dataset("dN", data=np.array(self.line_list['dN']))
            line_list.create_dataset("b", data=np.array(self.line_list['b']))
            line_list.create_dataset("db", data=np.array(self.line_list['db']))
            line_list.create_dataset("l", data=np.array(self.line_list['l']))
            line_list.create_dataset("dl", data=np.array(self.line_list['dl']))
            line_list.create_dataset("EW", data=np.array(self.line_list['EW']))
            line_list.create_dataset("Chisq", data=np.array(self.line_list['Chisq']))

    
    def plot_fit(self, ax=None, vel_range=600., filename=None):

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.velocities, self.fluxes, label='data', c='tab:grey', lw=2, ls='-')

        self.get_fluxes_model()
        for i in range(len(self.line_list)):
            p = np.array([self.line_list['N'][i], self.line_list['b'][i], self.line_list['l'][i]])
            _tau_model = pg.analysis.model_tau(self.ion_name, p, self.wavelengths)
            ax.plot(self.velocities, tau_to_flux(_tau_model), c='tab:pink', lw=1.5, ls='--')

        ax.plot(self.velocities, self.fluxes_model, label='model', c='tab:pink', ls='-', lw=2)

        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)
        for v in self.line_list['v']:
             ax.axvline(v, c='b', ls='--', lw=0.75)
        ax.set_xlim(self.gal_velocity_pos - vel_range, self.gal_velocity_pos +vel_range)
        ax.legend()
        if filename == None:
            filename = self.spectrum_file.replace('.h5', '.png')
        plt.savefig(filename)
        plt.clf()

