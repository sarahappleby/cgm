# Makefile for compiling Cloudy on a UNIX platform
# This works only with GNU make!
# you need g++ or pathCC to create Makefile.dep (set with CXXDEPEND)
# but after that you can use any compiler you want (set with CXX)

OPT = -O3 -ftrapping-math -fno-math-errno
DEBUGOPT = -finline
CXX = g++
# -W. is the old name for -Wextra, g++ 3.3 only supports the former
CXXFLAGS = -ansi ${OPT} -Wall -W -g
LDFLAGS = ${OPT} -Wall -g
LDLIBS =
NO_TRAP_FLAGS =
# this is intended for supplying extra command line parameters, e.g. extra -D flags
EXTRA =
SRCDIR = .
OBJEXT = o
LIBEXT = a

.PHONY: all default debug clean distclean depend

# Default rule comes first -- having this implement 'make all' is standard
all : default

# include a configuration file if one exists (or carry on without
# worrying if it doesn't)
-include Makefile.conf

main := runcloudylt_HM01_Eagle.cpp
source := ${notdir ${wildcard ${SRCDIR}/*.cpp}}
libsource := ${filter-out ${main},${source}}
includes := ${notdir ${wildcard ${SRCDIR}/*.h}}
objects := ${source:.cpp=.${OBJEXT}}
libobjects := ${libsource:.cpp=.${OBJEXT}}

ifneq (${SRCDIR},.)
  vpath %.cpp ${SRCDIR}
  vpath %.h   ${SRCDIR}
endif

# this is the compiler used to create Makefile.dep
CXXDEPEND = g++

# check compiler capabilities
CAPABILITIES := $(shell $(SRCDIR)/capabilities.pl $(CXX))
PRECOMPILE := $(filter precompile,$(CAPABILITIES))
VECTORIZE := $(filter vectorize,$(CAPABILITIES))

# DEP_GOALS will be an empty string for targets that don't need Makefile.dep
NODEP_GOALS = clean distclean depend
DEP_GOALS = ${filter-out ${NODEP_GOALS},${MAKECMDGOALS}}

# the default target does need Makefile.dep
ifeq (${MAKECMDGOALS},)
  DEP_GOALS = need_dependencies
endif

ifeq ($(VECTORIZE),vectorize)
  OPT += -ftree-vectorize
endif

DEBUG_GOALS = $(filter debug%,$(MAKECMDGOALS))
ifneq ($(DEBUG_GOALS),)
  OPT = $(DEBUGOPT)
endif

CXXDEFINES = ${filter -D%,${CXXFLAGS}}

# Precompiling the cddefines.h header speeds up compilation dramatically with g++
# versions from 3.4.0 onwards -- unfortunately, only one header can be precompiled.
# See http://gcc.gnu.org/onlinedocs/gcc-4.2.1/gcc/Precompiled-Headers.html

presource := cddefines.h
substname := cddefines.h
preobjects :=
ifeq ($(PRECOMPILE),precompile)
  substname := cddefines.h.gch
  preobjects := ${presource:.h=.h.gch}
endif

RANLIB := ${shell which ranlib}
ifneq ($(RANLIB),)
  RANLIB = ranlib
else
  RANLIB = ar ts
endif

PWD := ${shell pwd}

CLDCONFIG = $(PWD)/cloudyconfig.h

CDP = ${CLOUDY_DATA_PATH}
ifeq ($(CDP),)
# create reasonable default when CLOUDY_DATA_PATH not set...
  CDP = $(PWD)/${SRCDIR}/../data/
endif
ifneq (${CLOUDY_LAPACK_PATH},)
  CXXFLAGS += -DLAPACK -I${CLOUDY_LAPACK_PATH}/include
  LDLIBS += -L${CLOUDY_LAPACK_PATH}/lib -llapack
endif

# this is a collection of compiler flags that enable FPE traps
TRAP_FPE_FILTER = -fnonstd -ftrap=common -Ktrap=fp

SVNVER := ${shell which svnversion 2> /dev/null}
ifneq ($(SVNVER),)
  REVISION := ${shell $(SVNVER) $(SRCDIR)}
else
  REVISION = exported
endif
CXXFLAGS += -DSVN_REVISION=\"$(REVISION)\"

CXXFLAGS += -DSYS_CONFIG=\"$(CLDCONFIG)\"

# add in user-defined extra flags
CXXFLAGS += $(EXTRA)
LDFLAGS += $(EXTRA)

# this is a filter to erase optimization flags from CXXFLAGS
OPTFILTER = -O -O1 -O2 -O3 -O4 -O5 -Os -xO -xO1 -xO2 -xO3 -xO4 -xO5 /O /O1 /O2 /O3 -fast -xvector%
CFGFILTER = -DSYS_CONFIG%

CXXFLAGSNOOPT := ${filter-out ${OPTFILTER},${CXXFLAGS}}
CXXFLAGSNOCFG := ${filter-out ${CFGFILTER},${CXXFLAGS}}

# Makefile.targets can override or add to the DEFAULT variable.
DEFAULT = cloudy.exe

# include additional targets from file if one exists (or carry on without
# worrying if it doesn't)
-include Makefile.targets

debug : default

default : $(DEFAULT)

cloudy.exe : ${preobjects} maincl.${OBJEXT} libcloudy.${LIBEXT}
	${CXX} ${LDFLAGS} -o cloudy.exe maincl.${OBJEXT} -L. -lcloudy ${LDLIBS}

runcloudylt : ${preobjects} runcloudylt.o libcloudy.a
	${CXX} ${LDFLAGS} -o runcloudylt runcloudylt.${OBJEXT} -L. -lcloudy ${LDLIBS}

runcloudylt_HM01_Eagle : ${preobjects} runcloudylt_HM01_Eagle.o libcloudy.a
	${CXX} ${LDFLAGS} -o runcloudylt_HM01_Eagle runcloudylt_HM01_Eagle.${OBJEXT} -L. -lcloudy ${LDLIBS}

libcloudy.${LIBEXT}: ${libobjects}
	ar ru libcloudy.${LIBEXT} $^
	${RANLIB} libcloudy.${LIBEXT}

# Ensure version update if *any* source files change
version.o: ${main} ${libsource} ${includes}

clean :
	rm -f *.${OBJEXT}
	rm -f *.h.gch
	rm -rf SunWS_cache
	rm -f libcloudy.${LIBEXT}
	rm -f $(DEFAULT)

distclean :
	rm -f *.${OBJEXT}
	rm -f *.h.gch
	rm -f cloudyconfig.h
	rm -rf SunWS_cache
	rm -f libcloudy.${LIBEXT}
	rm -f $(DEFAULT)
	rm -f Makefile.dep
	rm -f ${SRCDIR}/Makefile.conf
	rm -rf lib
	rm -rf include

# ${source} ${includes}
depend Makefile.dep: $(CLDCONFIG)
	@echo "Updating dependency file, this may take a little while"
	@cd ${SRCDIR}; \
	${CXXDEPEND} -MM -DMM ${CXXDEFINES} ${source} | sed "s/${presource}/${substname}/" | sed "s/\.o/.${OBJEXT}/" > "$(PWD)/Makefile.dep"; \
	${CXXDEPEND} -MM -DMM ${CXXDEFINES} ${presource} | sed 's/\.o:/.h.gch:/' >> "$(PWD)/Makefile.dep";

cpu.${OBJEXT}:
	${CXX} ${CXXFLAGS} -c -DCLOUDY_DATA_PATH=\"${CDP}\" $<

# an explicit -O0 is needed for some compilers like icc
parse_crashdo.${OBJEXT}:
	${CXX} ${CXXFLAGSNOOPT} ${DEBUGOPT} -c $<

# this default rule is needed on Windows
%.${OBJEXT}: %.cpp
	${CXX} ${CXXFLAGS} -c -o $@ $<

%.h.gch: %.h
	${CXX} ${CXXFLAGS} -o $@ $<

$(CLDCONFIG): ${SRCDIR}/configure.sh
	@echo "Creating cloudyconfig.h"
	@${SRCDIR}/configure.sh ${CXX} ${CXXFLAGSNOCFG}

# only include Makefile.dep when we really need it, this prevents unnecessary builds of Makefile.dep
ifneq (${DEP_GOALS},)
-include Makefile.dep
endif
