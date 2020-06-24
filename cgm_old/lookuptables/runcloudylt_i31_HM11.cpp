/* runcloudylt.c -- Creates a lookup table of ionization fractions using cloudy 
   when given a ionization background, redshift, and fluxfactor 
   usage: runcloudylt infile, redshift, fluxfact > lookuptable
        * redshift = redshift of output
     * fluxfact = Factor by which amplitude should be *reduced*.
*/
#include "cddefines.h"
#include "cddrive.h"
#include "input.h"
#include "prt.h"
#include "save.h"
#include "monitor_results.h"
#include "grains.h"
#include "thirdparty.h"
#include "iontab.h"


#define float double

/*#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>*/
/*#define sqr(x) ((x)*(x))
#define max(A,B) ((A) > (B) ? (A) : (B))
#define min(A,B) ((A) < (B) ? (A) : (B))
#define mabs(A) ((A) < 0.0 ? -(A) : (A))*/
#define pi 3.1415926535898
#define NJMAX	1000
#define NINTERP	5000
#define Ryd	911.26708	/* 1 Rydberg in Angstroms */
#define NEXTRAP 20
#define NREDSHIFTS 60


int main(int argc,char **argv)

{
	int i, j, k, l;
	float jnuin[NJMAX][NJMAX];
	float wave[NJMAX],jnu[NJMAX],djnu[NJMAX],J[NJMAX];
	float redshift,fluxfac,J0,metal;
	float z[NJMAX];
	float lmin,lmax,jmin,jmax,oldwave;
	float Jx,Elam_keV;
	float hden, temp, fnu;
	float frac[51];
	int nz,iz,nl,nlam,nleft,lgOK;
	int xflag=0,cutflag=0,sfflag=0,H1cutflag=0;
	FILE *infile;
	FILE *outfile;
	char line[256], fname[256];

/* Parse input */
        if( (infile = fopen(argv[argc-3],"r")) == NULL ) {
                fprintf(stderr,"Could not open file %s\n",argv[argc-3]);
                exit(-1) ;
        }
	for( i=1; i<argc-3; i++ ) {
		if( strchr(argv[i],'h') != NULL || argv[i][0] != '-' ) {
			fprintf(stderr,"usage: makelt [-x] [-cut] [-H1cut] [-sf] [-h] HM_file redshift fluxfac\n");
			exit(-1);
		}
		if( strchr(argv[i],'x') != NULL ) xflag = 1;
		if( strstr(argv[i],"cut") != NULL ) cutflag = 1;
                if( strstr(argv[i],"H1cut") != NULL ) H1cutflag = 1;
		if( strstr(argv[i],"sf") != NULL ) sfflag = 1;
	}
	sscanf(argv[argc-2],"%lg",&redshift);
	sscanf(argv[argc-1],"%lg",&fluxfac);
	fprintf(stderr,"z=%g  ff=%g  flags: xray,cut,H1cut,sf=%d,%d,%d,%d\n",redshift,fluxfac,xflag,cutflag,H1cutflag,sfflag);

/* Input Haardt & Madau (2012) spectrum */
	nz = 0;
	for(iz=0;iz<NREDSHIFTS;iz++){
	  fscanf(infile,"%lf",&z[iz]);
	  nz++;
	}

	nl = 0;
	while(!feof(infile)){
	  fscanf(infile,"%lf",&wave[nl]);
	  for(iz=0;iz<NREDSHIFTS;iz++){
	    fscanf(infile,"%lf",&jnuin[nl][iz]);
	    //if(jnuin[nl][iz]<1e-50)jnuin[nl][iz]=1e-50;
	  }
	  nl++;
	}
	nl -= 1;

/* Handle dupicate wavelength entries */
	nlam = nl;
	nl = 0;
	for( i=0; i<nlam; i++ ) {
		if( wave[i] == oldwave ) wave[i] += 1.e-4*wave[i];
		wave[nl++] = wave[i];
		oldwave = wave[i];
	}
/* Find values of J(nu) spline-interpolated to the desired redshift */
	for( i=0; i<nl; i++ ) {
	  for( j=0; j<nz; j++ ) jnu[j] = jnuin[i][j];
	  spline(z,jnu,nz,1.e30,1.e30,djnu);
	  splint(z,jnu,djnu,nz,redshift,&J[i]);
	  J[i] /= fluxfac;
/* Add an X-ray background from Miyajo et al 98, if desired */
	  if( xflag ) {
	    Elam_keV = 12.4238/wave[i];
	    Jx = 6.626e-26*pow(Elam_keV,-0.42);
	    if( Jx > J[i] ) J[i] = Jx;
	  }
	  /* Do 4 Ry break, if desired */
	  if( cutflag ) if( wave[i]<0.25*Ryd ) J[i] *= 0.1;
	  if( H1cutflag ) if(wave[i]<1.00*Ryd ) J[i] *= 0.01;
	  //if(wave[i]>Ryd/14.7059) J[i] *= 5.5; /// HM12 low redshift mod!!!
	}

/* Generate filename of output lookup table and open file */
	sprintf(fname,"lt%02df%03d_i31",(int) (redshift*10.+0.5),(int) (fluxfac*100.+0.5));
	outfile = fopen(fname,"w");
	for( i=0; i<nl; i++ ) {
		if( wave[i]<Ryd && wave[i+1]>Ryd ) {
			J0 = J[i]+(J[i+1]-J[i])*(Ryd-wave[i])/(wave[i+1]-wave[i]);
			break;
		}
	}
	
	/* Is this proper treatment of metallicity, or does it not matter that much? */
	//	metal = -2.5;
	//	if( redshift < 2. ) metal = -2.0;
	metal = -3.0;
	fnu = log10(4*pi*J0);

	fprintf(stderr,"nz=%d nl=%d metal=%g J0=%g outfile=%s\n",nz,nl,metal,J0,fname);

/* Convert to log-log space for doing interpolations */
	for( i=0; i<nl; i++ ) {
	  fprintf(stdout,"%g %g\n",wave[i],J[i]);
		wave[i] = log10(wave[i]);
		if(J[i]<1e-30) J[i] = 1e-30;
		J[i] = log10(J[i]);
	}
/* Do loglinear extrapolation to CLOUDY frequency limits */
	J0 = log10(J0);
	lmin=log10(1.239e-04);
        lmax=log10(9.1127e+07);
	jmin=J[NEXTRAP-1]+(J[NEXTRAP-1]-J[0])*(lmin-wave[NEXTRAP-1])/(wave[NEXTRAP-1]-wave[0]);
        jmax=J[nl-1]+(J[nl-1]-J[nl-NEXTRAP-1])*(lmax-wave[nl-1])/(wave[nl-1]-wave[nl-NEXTRAP-1]);
	fprintf(stderr,"%g %g %g %g %g\n",Ryd/pow(10.,lmin),Ryd/pow(10.,lmax),jmin-J0,jmax-J0,J0);

/* Add value to beginning of wave, J list */
	for( i=nl-1; i>=0; i-- ) {
		wave[i+1] = wave[i];
		J[i+1] = J[i];
	}
	wave[0] = lmin;
	J[0] = jmin;
	nl++;
/*	for( i=0; i<nl; i++ ) fprintf(stdout,"%d %g %g %g %g\n",i,pow(10.,wave[i]),J[i],Ryd/pow(10.,wave[i]),J[i]-J0);
	exit(0);*/

/* Convert back from log-log space */
	for( i=0; i<nl; i++ ) {
		wave[i] = pow(10.,wave[i]);
		J[i] = pow(10.,J[i]);
	}
	lmin = pow(10.,lmin);
	lmax = pow(10.,lmax);

	for(i=1;i<=NHPTS;i++){
	  hden = NHLOW + (i-1)*DELTANH;
	  for(j=1;j<=TPTS;j++){
	    temp= TLOW + (j-1)*DELTAT;
	    printf("J0 = %5.3e\n",J0);
	    cdInit();
	    cdTalk(1);
	    sprintf(line,"set dr 0");
	    nleft = cdRead(line);
	    //sprintf(line, "element helium abundance -1.1");
	    sprintf(line, "element helium abundance -1");
	    nleft = cdRead(line);
	    sprintf(line,"metals %5.2f ",metal);
	    nleft = cdRead(line);
	    sprintf(line,"interp (    0    %.5f)",jmax-J0);
	    nleft = cdRead(line);
	    for( k=nl-1; k>=0; k-=2 ) {
	      if( Ryd/wave[k-1]>1000 ) sprintf(line,"cont (%.6f %.6f) (%.6f %.6f)",Ryd/wave[k],log10(J[k])-J0,Ryd/wave[k-1],log10(J[k-1])-J0);
	      else if( Ryd/wave[k-1]>1 ) sprintf(line,"cont (%.6f %.6f) (%.6f %.6f)",Ryd/wave[k],log10(J[k])-J0,Ryd/wave[k-1],log10(J[k-1])-J0);
	      else sprintf(line,"cont (%.6f %.6f) (%.6f %.6f)",Ryd/wave[k],log10(J[k])-J0,Ryd/wave[k-1],log10(J[k-1])-J0);
	      fprintf(stdout,"%s\n",line);
	      nleft = cdRead(line);	      
	    }
	    sprintf(line, "F(nu) = %8.4f ",fnu);
	    nleft = cdRead(line);
	    sprintf(line, "no molecules");
	    nleft = cdRead(line);
	    sprintf(line, "constant temperature = %7.4f",temp);
	    nleft = cdRead(line);
	    sprintf(line, "hden = %7.4f",hden);
	    nleft = cdRead(line);
	    sprintf(line, "stop zone 1");
	    nleft = cdRead(line);
	    //sprintf(line, "stop column density 3");
	    //nleft = cdRead(line);
	    //sprintf(line, "case A");
	    //nleft = cdRead(line);
	    //sprintf(line, "case B");
	    //nleft = cdRead(line);
	    //sprintf(line, "set dielectric recombination Badnell");
	    //nleft = cdRead(line);
	    //sprintf(line, "set dielectronic recombination Burgess off");
	    //nleft = cdRead(line);
            //sprintf(line, "set dielectronic recombination Nussbaumer off");
            //nleft = cdRead(line);
            //sprintf(line, "set dielectronic recombination kludge");
            //nleft = cdRead(line);
	    //sprintf(line, "no charge transfer");
	    //nleft = cdRead(line);
	    //sprintf(line, "no Auger effect");
	    //nleft = cdRead(line);
	    //sprintf(line, "no diffuse line pumping");
	    //nleft = cdRead(line);
	    //sprintf(line, "no FeII pumping");
	    //nleft = cdRead(line);
	    //sprintf(line, "no fine structure line optical depths");
	    //nleft = cdRead(line);
	    //sprintf(line, "no fine opacitites");
	    //nleft = cdRead(line);
	    //sprintf(line, "no grain physics");
	    //nleft = cdRead(line);
	    //sprintf(line, "no induced processes");
	    //nleft = cdRead(line);
	    //sprintf(line, "no Lya 21cm pumping");
	    //nleft = cdRead(line);
	    //sprintf(line, "no secondary ionizations");
	    //nleft = cdRead(line);
	    //sprintf(line, "no three body recombination");
	    //nleft = cdRead(line);
	    //sprintf(line, "no UTA ionization");
	    //nleft = cdRead(line);
	    //sprintf(line, "atom H-like collisions off");
	    //nleft = cdRead(line);
	    sprintf(line, "cmb redshift = %g",redshift);
	    nleft = cdRead(line);
	    sprintf(line, "iterate to convergence");
	    nleft = cdRead(line);
	    if(cdDrive()>0)fprintf(stderr,"DIDN'T WORK\n");
	    k = 1;
	    printf("calculating!!!\n");
	    lgOK = cdIonFrac("hydr", 1, &frac[1], "VOLUME", 0);
	    lgOK = cdIonFrac("heli", 1, &frac[2], "VOLUME", 0);
	    lgOK = cdIonFrac("heli", 2, &frac[3], "VOLUME", 0);
	    lgOK = cdIonFrac("carb", 2, &frac[4], "VOLUME", 0);
	    lgOK = cdIonFrac("carb", 3, &frac[5], "VOLUME", 0);
	    lgOK = cdIonFrac("carb", 4, &frac[6], "VOLUME", 0);
	    lgOK = cdIonFrac("carb", 5, &frac[7], "VOLUME", 0);
	    lgOK = cdIonFrac("nitr", 4, &frac[8], "VOLUME", 0);
	    lgOK = cdIonFrac("nitr", 5, &frac[9], "VOLUME", 0);
	    lgOK = cdIonFrac("nitr", 6, &frac[10], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 1, &frac[11], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 2, &frac[12], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 3, &frac[13], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 4, &frac[14], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 5, &frac[15], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 6, &frac[16], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 7, &frac[17], "VOLUME", 0);
	    lgOK = cdIonFrac("oxyg", 8, &frac[18], "VOLUME", 0);
	    lgOK = cdIonFrac("neon", 4, &frac[19], "VOLUME", 0);
	    lgOK = cdIonFrac("neon", 8, &frac[20], "VOLUME", 0);
	    lgOK = cdIonFrac("magn", 2, &frac[21], "VOLUME", 0);
	    lgOK = cdIonFrac("magn", 10, &frac[22], "VOLUME", 0);
	    lgOK = cdIonFrac("alum", 2, &frac[23], "VOLUME", 0);
	    lgOK = cdIonFrac("alum", 3, &frac[24], "VOLUME", 0);
	    lgOK = cdIonFrac("sili", 2, &frac[25], "VOLUME", 0);
	    lgOK = cdIonFrac("sili", 3, &frac[26], "VOLUME", 0);
	    lgOK = cdIonFrac("sili", 4, &frac[27], "VOLUME", 0);
	    lgOK = cdIonFrac("sili", 12, &frac[28], "VOLUME", 0);
	    lgOK = cdIonFrac("carb", 6, &frac[29], "VOLUME", 0);
	    lgOK = cdIonFrac("neon", 9, &frac[30], "VOLUME", 0);
	    //lgOK = cdIonFrac("phos", 4, &frac[29], "VOLUME", 0);
	    //lgOK = cdIonFrac("sulp", 6, &frac[30], "VOLUME", 0);
	    lgOK = cdIonFrac("iron", 2, &frac[31], "VOLUME", 0);

	    for(l=1;l<=31;l++){
	      frac[l] = log10(frac[l]);
	      if(frac[l] < -9.99999) frac[l] = -9.99999;
	      fprintf(outfile,"%9.5f ",frac[l]);
	      printf("%9.5f ",frac[l]);
	    }
	    fprintf(outfile,"\n");
	    printf("\n");	    
	  }
	}
	fclose(outfile);
	exit(0);
}

