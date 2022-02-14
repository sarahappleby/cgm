from astropy.convolution import convolve, Gaussian1DKernel

pixel_size = 2.5 # km/s
fwhm = 6. / pixel_size # get fwhm from km/s into pixels
gauss_kernel = Gaussian1DKernel(stddev=fwhm / 2.355)

plt.plot(gauss_kernel)
plt.show()
