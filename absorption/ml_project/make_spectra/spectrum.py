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

    def get_initial_window(self, vel_range, v_central=None, v_boxsize=10000.):

        def _find_nearest(array, value):
            return np.abs(array - value).argmin()

        if v_central is None:
            v_central = self.gal_velocity_pos

        # get the velocity start and end positions
        dv = self.velocities[1] - self.velocities[0]
        v_start = v_central - vel_range
        v_end = v_central + vel_range
        N = int((v_end - v_start) / dv)

        # get the start and end indices
        if v_start < 0.:
            v_start += v_boxsize
        i_start = _find_nearest(self.velocities, v_start)
        i_end = i_start + N

        return i_start, i_end

    def extend_to_continuum(self, i_start, i_end, contin_level=1.):

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_start, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.05:
                i_start -= 1
            else:
                continuum = True

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_end, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.05:
                i_end += 1
            else:
                continuum = True

        return i_start, i_end
    
    def buffer_with_continuum(self, waves, flux, noise, nbuffer=50,):

        dl = waves[1] - waves[0]
        l_start = np.arange(waves[0] - dl*nbuffer, waves[0], dl)
        l_end = np.arange(waves[-1]+dl, waves[-1] + dl*(nbuffer+1), dl)
        
        waves = np.concatenate((l_start, waves, l_end))

        sigma_noise = 1./snr
        new_noise = np.random.normal(0.0, sigma_noise, 2*nbuffer)
        noise = np.concatenate((new_noise[:nbuffer], noise, new_noise[nbuffer:]))
        flux = np.concatenate((tau_to_flux(np.zeros(nbuffer)) + new_noise[:nbuffer], flux, tau_to_flux(np.zeros(nbuffer)) + new_noise[nbuffer:]))
        
        return waves, flux, noise 


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
        self.fluxes_model = tau_to_flux(self.tau_model)

   
    def run(self, vel_range):
    
        i_start, i_end = get_initial_window(vel_range) 
        i_start, i_end = extend_to_continuum(i_start, i_end)

        if i_start < 0:
            i_start += len(self.wavelengths)
            i_end += len(self.wavelengths)

        waves = self.wavelengths.take(range(i_start, i_end), mode='wrap')
        flux = self.fluxes.take(range(i_start, i_end), mode='wrap')
        noise = self.noise.take(range(i_start, i_end), mode='wrap')

        # check if the start and end wavelengths go over the limits of the box
        i_wrap = len(self.wavelengths) - i_start
        if i_wrap < N:
            # spectrum wraps, i_wrap is the first index of the wavelengths that have been moved to the left side of the box
            dl = self.wavelengths[1] - self.wavelengths[0]
            wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
            waves[i_wrap:] += wave_boxsize + dl
            # then for any fitted lines with position outwith the right-most box limits: subtract dl + wave_boxsize

        # buffer to continuum

        # fit spectra

        # adjust the output lines to cope with wrapping


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

