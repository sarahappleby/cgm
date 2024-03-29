# Classily written spectrum fitting routines :) 

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pygad as pg
from physics import wave_to_vel, vel_to_wave, tau_to_flux
from utils import read_h5_into_dict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

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

        # get the portion of the CGM spectrum that we want to fit

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

        return i_start, i_end, N


    def extend_to_continuum(self, i_start, i_end, N, contin_level=None):

        # from the initial velocity window, extend the start and end back to the level of the continuum of the input spectrum/

        if contin_level is None:
            contin_level = self.continuum[0]

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_start, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.02:
                i_start -= 1
                N += 1
            else:
                continuum = True

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_end, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.02:
                i_end += 1
                N += 1
            else:
                continuum = True

        return i_start, i_end, N
   

    def buffer_with_continuum(self, waves, flux, nbuffer=50, snr_default=30.):

        # add a buffer to either end of the velocity window at the continuum level to aid the voigt fitting.

        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        dl = waves[1] - waves[0]
        l_start = np.arange(waves[0] - dl*nbuffer, waves[0], dl)
        l_end = np.arange(waves[-1]+dl, waves[-1] + dl*(nbuffer+1), dl)
        
        waves = np.concatenate((l_start, waves, l_end))

        sigma_noise = 1./snr
        new_noise = np.random.normal(0.0, sigma_noise, 2*nbuffer)
        flux = np.concatenate((tau_to_flux(np.zeros(nbuffer)) + new_noise[:nbuffer], flux, tau_to_flux(np.zeros(nbuffer)) + new_noise[nbuffer:]))
        
        return waves, flux

    def prepare_spectrum(self, vel_range, do_continuum_buffer=True, nbuffer=50, snr_default=30):

        # cut out the portion of the spectrum that we want within some velocity range, making sure the section we cut out 
        # goes back up to the conintuum level (no dicontinuities)

        print('getting initial window')
        i_start, i_end, N = self.get_initial_window(vel_range)
        print(i_start, i_end)
        i_start, i_end, N = self.extend_to_continuum(i_start, i_end, N)
        print(i_start, i_end)

        # cope with spectra that go beyond the left hand edge of the box (periodic wrapping)
        if i_start < 0:
            i_start += len(self.wavelengths)
            i_end += len(self.wavelengths)

        # extract the wavelengths and fluxes for fitting
        self.waves_fit = self.wavelengths.take(range(i_start, i_end), mode='wrap')
        self.fluxes_fit = self.fluxes.take(range(i_start, i_end), mode='wrap')

        # check if the start and end wavelengths go over the limits of the box
        i_wrap = len(self.wavelengths) - i_start
        wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
        dl = self.wavelengths[1] - self.wavelengths[0]
        if i_wrap < N:
            # spectrum wraps, i_wrap is the first index of the wavelengths that have been moved to the left side of the box
            self.waves_fit[i_wrap:] += wave_boxsize + dl
            # then for any fitted lines with position outwith the right-most box limits: subtract dl + wave_boxsize

        # add a buffer of continuum to either side to help the voigt fitter identify where to fit
        print('Doing continuum buffer')
        if do_continuum_buffer is True:
            self.waves_fit, self.fluxes_fit = self.buffer_with_continuum(self.waves_fit, self.fluxes_fit, nbuffer=nbuffer)

        # get the noise level
        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        self.noise_fit = np.asarray([1./snr] * len(self.fluxes_fit))


    def fit_spectrum_old(self, vel_range=600., nbuffer=20):

        # old fitting routine - not required

        contin_level = self.continuum[0]
        self.extend_to_continuum(vel_range, contin_level)

        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths[self.vel_mask], self.fluxes[self.vel_mask], self.noise[self.vel_mask],
                                                  chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
    
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)


    def fit_periodic_spectrum(self):

        # the fitting approach for periodic spectra, i.e. those which span the length of the Simba volume

        wrap_flux, wrap_noise, wrap_start = pg.analysis.periodic_wrap(self.wavelengths, self.fluxes, self.noise)
        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths, wrap_flux, wrap_noise,
                                         chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
        self.line_list['l'] = pg.analysis.periodic_unwrap_wavelength(self.line_list['l'], self.wavelengths, wrap_start)
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)


    def get_tau_model(self):

        # compute the total optical depth of the model from the individual lines

        self.tau_model = np.zeros(len(self.wavelengths))
        for i in range(len(self.line_list["N"])):
            p = np.array([self.line_list["N"][i], self.line_list["b"][i], self.line_list["l"][i]])
            self.tau_model += pg.analysis.model_tau(self.ion_name, p, self.wavelengths)

    
    def get_fluxes_model(self):

        # compute the total flux from the individual lines

        self.get_tau_model()
        self.fluxes_model = tau_to_flux(self.tau_model)


    def write_line_list(self):

        # save the components of the fit in h5 format to the original input file

        with h5py.File(self.spectrum_file, 'a') as hf:
            if 'line_list' in hf.keys():
                del hf['line_list']
            elif 'lines' in hf.keys():
                del hf['lines']
            line_list = hf.create_group("line_list")
            for k in self.line_list.keys():
                line_list.create_dataset(k, data=np.array(self.line_list[k]))


    def plot_fit(self, ax=None, vel_range=600., filename=None):

        # plot the results :)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.velocities, self.fluxes, label='data', c='tab:grey', lw=2, ls='-')

        self.get_fluxes_model()
        for i in range(len(self.line_list['N'])):
            p = np.array([self.line_list['N'][i], self.line_list['b'][i], self.line_list['l'][i]])
            _tau_model = pg.analysis.model_tau(self.ion_name, p, self.wavelengths)
            ax.plot(self.velocities, tau_to_flux(_tau_model), c='tab:pink', alpha=0.5, lw=1, ls='--')

        ax.plot(self.velocities, self.fluxes_model, label='model', c='tab:pink', ls='-', lw=2)

        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(self.gal_velocity_pos - vel_range, self.gal_velocity_pos +vel_range)
        ax.legend()
        
        chisq = np.around(np.unique(self.line_list['Chisq']), 2)
        chisq = [str(i) for i in chisq]
        plt.title(r'$\chi^2_r = {x}$'.format(x = ', '.join(chisq) ))
        
        if filename == None:
            filename = self.spectrum_file.split('/')[-1].replace('.h5', '.png')
        plt.savefig(filename)
        plt.close()


    def main(self, vel_range, do_continuum_buffer=True, nbuffer=50, 
             snr_default=30., chisq_unacceptable=25, chisq_asym_thresh=-3., 
             do_prepare=True, do_regions=False, do_fit=True, write_lines=False, plot_fit=False):
  
        # prepare the portion of the spectrum to fit
        # extract from full spectrum, wrap periodically, buffer with a continuum, set the noise level for fitting
        if do_prepare:
            self.prepare_spectrum(vel_range, do_continuum_buffer=True, nbuffer=50, snr_default=30.,)

        # to identify the region boundaries only:
        if do_regions:
            if do_prepare is not True:
                print('Spectrum not prepared; set do_prepare=True and retry :)')
                return
            else:
                self.line_list = {}
                self.regions_l, self.regions_i = pg.analysis.find_regions(self.waves_fit, self.fluxes_fit, self.noise_fit, min_region_width=2, extend=True)
                self.line_list['region'] = np.arange(len(self.regions_l))

        # to perform the voigt fitting:
        if do_fit:
            print('Fitting...')
            if self.ion_name == 'H1215':
                logN_bounds = [12, 19]
            else:
                logN_bounds = [11, 17]
            b_bounds=None
         
            self.line_list = pg.analysis.fit_profiles(self.ion_name, self.waves_fit, self.fluxes_fit, self.noise_fit,
                                                      chisq_lim=2.5, chisq_unacceptable=chisq_unacceptable, 
                                                      chisq_asym_thresh=chisq_asym_thresh, 
                                                      max_lines=10, logN_bounds=logN_bounds, 
                                                      b_bounds=b_bounds, mode='Voigt')
       
            # adjust the output lines to cope with wrapping
            for i in range(len(self.line_list['l'])):
                if self.line_list['l'][i] > self.wavelengths[-1]:
                    self.line_list['l'][i]  -= (wave_boxsize + dl)
                elif self.line_list['l'][i] < self.wavelengths[0]:
                    self.line_list['l'][i] += (wave_boxsize + dl)

        # keep the fit, save to the original spectrum file
        print(self.line_list)
        if write_lines:
            self.write_line_list()

        if plot_fit:
            self.plot_fit()


if __name__ == '__main__':
    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    fr200 = 0.25
    line = 'H1215'
    orient = 0
    vel_range = 600.
    chisq_asym_thresh = -3.
    
    spectrum_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/'
    spectrum_file = f'{spectrum_dir}sample_galaxy_195_{line}_{orient}_deg_{fr200}r200.h5'

    spec = Spectrum(spectrum_file)
    spec.main(vel_range=vel_range, chisq_asym_thresh=chisq_asym_thresh, plot_fit=True)
