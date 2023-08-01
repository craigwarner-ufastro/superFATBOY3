# RCS: 
# $Id: Makefile,v 1.1 2016/07/13 21:13:11 warner Exp $
#
# Macros:

SHELL := /bin/tcsh -f

JAVAC = javac
JFLAGS = -classpath javaUFProtocol.jar:javaUFLib.jar -d ./ -deprecation

JPKG := sfbPlot
JMAIN:= SuperFatboyPlot

JSRC =	$(JMAIN).java  UFRSSPlot.java

JCLASS := $(patsubst %.java, $(JPKG)/%.class, $(JSRC))

RCSLIST = Makefile  $(JSRC) $(JMAIN)

#targets:

all: install

$(JPKG)/%.class: $(JSRC)
	echo 'test'
	$(JAVAC) $(JFLAGS) $<

install: build $(JPKG).jar
	-@echo installed SuperFatboyPlot 

build: $(JSRC)
	$(JAVAC) $(JFLAGS) $(JSRC)

$(JPKG).jar: $(JCLASS)
	echo 'Main-Class: $(JPKG).$(JMAIN)' >! manifest
	echo 'Class-Path: $(JPKG).jar javaUFProtocol.jar javaUFLib.jar' >> manifest
	cp -f -p gator_small.gif $(JPKG)
	jar cmf manifest $@ $(JPKG) 

show:
	-@echo JAVAC = $(JAVAC)
	-@echo JFLAGS =$(JFLAGS)
	-@echo RCSLIST = $(RCSLIST)
	-@echo JSRC = $(JSRC)
	-@echo JCLASS = $(JCLASS)
	-@echo JPKG = $(JPKG)

coall:
	co RCS/*,v
	if ( ! -e $(JPKG) ) mkdir -p $(JPKG)

init: $(RCSLIST)
	if ( ! -e $(JPKG) ) mkdir -p $(JPKG)
	co RCS/*

clean:
	-$(RM) -r $(JPKG)  *.jar  *.class
