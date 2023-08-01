# RCS: 
# $Id: Makefile,v 1.3 2022/05/14 14:53:18 warner Exp $
#
# Macros:
#.IGNORE:
#       @true

#include ../.makerules

SHELL := /bin/tcsh -f

DOMAIN := $(shell domainname)

#reset RCSMASTER & DEV:
RCSDEV := $(shell \pwd)
RCSMASTER := $(shell \ls -l | \grep RCS | \cut -d'>' -f2 | \cut -c2- | sed 's^/RCS*^^')

UFDIRS := config linelists templates 

RCSLIST := Makefile $(UFDIRS)

# Targets:
init: initco
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k init; popd;)

initco: 
	$(foreach i, $(UFDIRS), if( ! -e $i ) mkdir $i;)
	$(foreach i, $(UFDIRS), pushd $i; if ( ! -e RCS && -e $(RCSMASTER) ) ln -s $(RCSMASTER)/$i/RCS; popd;)
	$(foreach i, $(UFDIRS), pushd $i; if( -e RCS ) co -q RCS/*,v; popd;)

clean:
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k clean; popd;)

#include ../.maketargets
