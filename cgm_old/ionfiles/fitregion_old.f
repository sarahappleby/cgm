	subroutine fitregion
c
c  fits Voigt profiles to spectrum
c
	include 	'vpdefs.h'
	integer i,imin,imax,j
	integer subreg(maxregions,2),nsubreg,isubreg
	double precision fmin,fminreg
	integer lineflag(maxregions)
	integer ew_finder
c
	nlines = 0
c  loop until all detection regions processed
c  find regions where lines are detected above N_sigma significance
	do 35 while(ew_finder(1,ndata,subreg,nsubreg,resid).ne.0)
c  find region with lowest min flux
	  isubreg = 1
	  fmin = resid(subreg(1,1))
	  do 40 i=1,nsubreg
	    fminreg = resid(subreg(i,1))
	    do 45 j=subreg(i,1),subreg(i,2)
	      fminreg = MIN(fminreg,resid(j))
 45	    continue
	    if (fminreg.lt.fmin) then
	      isubreg = i
	      fmin = fminreg
	    end if
 40	  continue
c  set up region for fitting
	  nlines = nlines + 1
	  imin = subreg(isubreg,1)
	  imax = subreg(isubreg,2)
	  region(nlines,1) = imin
	  region(nlines,2) = imax
c  smooth data
	  CALL smooth(imin,imax)
c  find minimium flux point in region 
	  CALL findmin(imin,imax)
c  fit line at minimum flux point
C	do 2 j=-3,3
C	  k = j+centpix(nlines)
C	  write(6,*) k,vel(k),resid(k)
C 2	continue
	  if (minflux(centpix(nlines)).gt.
     &	    fsigma*noise(centpix(nlines))) then
	    write(6,'(a,3i8,3f12.4)') 'min at ',centpix(nlines),
     &	    imin,imax,
     &	    vel(centpix(nlines)),
     &	    minflux(centpix(nlines)),noise(centpix(nlines))
	    CALL fitline(nlines,imin,imax)
	  else
	    write(6,'(a,3i8,3f12.4)') 'sat at ',centpix(nlines),
     &	    imin,imax,
     &	    vel(centpix(nlines)),
     &	    minflux(centpix(nlines)),noise(centpix(nlines))
C	    CALL polyfit(imin,imax)
	    CALL fitsat(nlines,imin,imax)
	  end if
	  write(6,'(a,2i10,3f14.6)') '  fit line: ',nlines,
     &    centpix(nlines),
     &	  vel(centpix(nlines)),NHI(nlines)/1.d13,bpar(nlines)
	  flush(6)
 35	continue

c  if metal, search for matching doublet, discard doublet line
	if( fdoublet.ne.0.d0.AND.lambda1.ne.0.d0 ) then
C	  CALL doublet(lineflag)
	endif

	return
	end

C=============================================================================

	subroutine findmin(imin,imax)

	include 'vpdefs.h'
	integer imin,imax,i
	double precision fluxtemp

	fluxtemp = minflux(imin)
	centpix(nlines) = imin
	do 10 i=imin+1,imax
	  if (minflux(i).lt.fluxtemp) then
	    centpix(nlines) = i
	    fluxtemp = minflux(i)
	  endif
 10	continue
	if (imax-imin.ge.4) then
	  if (centpix(nlines).ge.imax-1) centpix(nlines) = imax-2
	  if (centpix(nlines).le.imin+1) centpix(nlines) = imin+2
	endif

c  find max noise in region; set fsigma 
c       avenoise = 0.d0
c        do 25 i=imin,imax
c          avenoise = avenoise+noise(i)
c 25     continue
c       avenoise = avenoise/(imax-imin)
        avenoise = noise(imin)
        do 5 i=imin,imax
          avenoise = MAX(avenoise,noise(i))
 5      continue

	return
	end

C=============================================================================

	subroutine fitline(n,imin,imax)

	include 'vpdefs.h'
	integer n,imin,imax
	logical shifted
	integer modelbelow

        do 7 i=imin,imax
	  if (minflux(i).gt.2.*fsigma*avenoise) then
            minflux(i) = minflux(i) - fsigma*avenoise
	  else
	    minflux(i) = 0.5*minflux(i)
	  end if
 7      continue
c  first guess is HUGE
	NHI(n) = NHImaxns
	bpar(n) = MIN(bparmax,vel(imax)-vel(imin))
	CALL model(n,n,imin,imax,1)	! determines workflux
C	write(6,*) imin,imax,vel(imin),vel(imax),bpar(n)
c  reduce NHI until value at n is above minimum acceptable level
	do 10 while (workflux(centpix(n)).lt.minflux(centpix(n)))
	  NHI(n) = freduce*NHI(n)
	  CALL model(n,n,imin,imax,1)	! determines workflux
 10	continue
c     	write(6,'(a5,3i6,6f9.1)') 'b: ',centpix(n),imin,imax,
c     &		vel(imax),vel(imin),
c     &		NHI(n)/1.e13,bpar(n),
c     &		workflux(centpix(n)),minflux(centpix(n))
c     	write(6,*) 'adjusted NHI: ',NHI(n)/1.e13,workflux(centpix(n)),
c     &		minflux(centpix(n))
c  reduce b (simultaneously keeping NHI at n above minflux)
c  until model lies entirely above minflux
	do 20 while (modelbelow(imin,imax,n).ne.0)
c  if one side is completely above, see if we can shift central vel
	  if (modelbelow(centpix(n),imax,n).eq.0.AND..NOT.shifted
     &	    .AND.centpix(n).lt.imax-1) then
	    centpix(n) = centpix(n)+1
	    shifted = .TRUE.
	  else if (modelbelow(imin,centpix(n),n).eq.0.AND..NOT.shifted
     &	    .AND.centpix(n).gt.imin+1) then
	    centpix(n) = centpix(n)-1
	    shifted = .TRUE.
c  if not, reduce b
	  else
	    bpar(n) = freduce*bpar(n)
c	    if( ABS(vel(centpix(n))-vel(centpix(n)-1)).lt.bpar(n) ) 
	    shifted = .FALSE.
	  endif
	  CALL model(n,n,imin,imax,1)	! determines workflux
	  do 30 while (workflux(centpix(n)).lt.minflux(centpix(n)))
	    NHI(n) = freduce*NHI(n)
	    CALL model(n,n,imin,imax,1)	! determines workflux
 30	  continue
 20	continue

c  compute residual flux
	CALL model(1,n,1,ndata,1)
	do 80 i=imin,imax
          residold = resid(i)
          resid(i) = 1.d0 + flux(i) - workflux(i)
c       write(6,*) i,resid(i),residold
	  if (resid(i).gt.1.d0) resid(i) = 1.d0
          noise(i) = noise(i)*sqrt(MAX(resid(i),noise(i))/
     &      (MAX(residold,noise(i))))
 80	continue

	return 
	end

C=============================================================================

	integer function modelbelow(imin,imax,n)

	include 'vpdefs.h'
	integer imin,imax,n,i
	double precision fracbelow

	modelbelow = 0
	if (imax.eq.imin) return
	do 10 i=imin,imax
	  if (workflux(i).lt.minflux(i)) modelbelow = modelbelow+1
c	  if (workflux(i).lt.minflux(i).AND.imax.ne.centpix(n).
c     &  AND.imin.ne.centpix(n)) 
c     &	write(6,'(i10,3f12.3)') i,vel(i),workflux(i),minflux(i)
 10	continue
	fracbelow = 1.d0*modelbelow/(imax-imin)
	if (fracbelow.lt.0.05.OR.modelbelow.le.1) modelbelow = 0

	return
	end

