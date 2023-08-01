package sfbPlot;

/**
 * Title:        UFRSSPlot.java
 * Version:      (see rcsID)
 * Copyright:    Copyright (c) 2005
 * Author:       Craig Warner
 * Company:      University of Florida
 * Description:  UFPlotPanel for flam2helper
 */

import javaUFLib.*;
import javaUFProtocol.*;

import java.awt.*;
import javax.swing.*;
import java.awt.image.*;
import java.awt.geom.*;
import java.awt.event.*;
import java.util.*;
import java.io.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.awt.event.*;
import javax.swing.border.*;

public class UFRSSPlot extends UFPlotPanel implements ActionListener {
    public static final
        String rcsID = "$Name:  $ $Id: UFRSSPlot.java,v 1.6 2019/09/20 19:44:35 warner Exp $";

    protected int numSpectra = 0;
    protected String title = "", plotOpts = "", addOpts = "";
    protected JMenuItem resetRangeItem;
    protected ArrayList<SuperFatboySpectrum> spectra;
    protected ArrayList<RSSContainer> rssContainers;

    long currTime;
    SuperFatboyPlot.UFRSSPlotPanel plotPanel = null;

    protected String[] dragObjectStatus = {"Hidden", "Hidden", "Hidden", "Hidden"};
    protected Color[] dragObjectColor = {Color.RED, Color.ORANGE, Color.GREEN, Color.BLUE};
    protected int[] dragObjectX = {100, 150, 200, 250};
    protected int[] dragObjectY = {100, 150, 200, 250};
    protected boolean[] dragObject = {false, false, false, false};
    protected int mouseButton = 0;
    protected int xisav, yisav, xInit = 0, yInit = 0;
    protected boolean isOldLog = false;
    int pos = 0;

    private float[] x;
    private float[] tempy;

    public UFRSSPlot() {
      numSpectra = 0;
      spectra = new ArrayList();
      rssContainers = new ArrayList();
      setupPlot(640,512);
    }

    public UFRSSPlot(String fitsFile) {
      this();
      addFitsFile(fitsFile);
    }

    public void addFitsFile(String fitsFile) {
      UFFITSheader imgFITShead = new UFFITSheader();
      Vector<UFProtocol> data = imgFITShead.readFITSmef(fitsFile);
      int nframes = data.size();
      int rows = imgFITShead.getLastFrameCon().height;
      int cols = imgFITShead.getLastFrameCon().width;
      UFFloats frameData = castAsUFFloats(data.elementAt(0));
      float[] ys = frameData.values();
      int pos = numSpectra; //record current pos
      for (int j = 0; j < rows; j++) {
	spectra.add(new SuperFatboySpectrum(imgFITShead, ys, j));
	numSpectra++;
      }
      rssContainers.add(new RSSContainer(fitsFile, pos, numSpectra-pos, this));
    }

    public void updatePlotOpts(String plotOpts, String[] oPlotOpts) {
      this.plotOpts = plotOpts;
      for (int j = 0; j < oPlotOpts.length; j++) {
	if (j < spectra.size()) spectra.get(j).oplotOpts = oPlotOpts[j];
      }
    }

    public void setPlotPanel(SuperFatboyPlot.UFRSSPlotPanel plotPanel) {
      this.plotPanel = plotPanel;
      if (!resetRangeItem.isVisible()) resetRangeItem.setVisible(true);
    }

    public void setupPlot(int xdim, int ydim) {
      this.xdim = xdim;
      this.ydim = ydim;
      xpos2 = xdim - 20;
      ypos2 = ydim - 47;
      setBackground(Color.black);
      setForeground(Color.white);
      setPreferredSize(new Dimension(xdim, ydim));

      popMenu = new JPopupMenu();

      exportItem = new JMenuItem("Export as PNG/JPEG");
      popMenu.add(exportItem);
      exportItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ev) {
           JFileChooser jfc = new JFileChooser(saveDir);
           int returnVal = jfc.showSaveDialog((Component)ev.getSource());
           if (returnVal == JFileChooser.APPROVE_OPTION) {
              String filename = jfc.getSelectedFile().getAbsolutePath();
              saveDir = jfc.getCurrentDirectory();
              File f = new File(filename);
              if (f.exists()) {
                String[] saveOptions = {"Overwrite","Cancel"};
                int n = JOptionPane.showOptionDialog(UFRSSPlot.this, filename+" already exists.", "File exists!", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE, null, saveOptions, saveOptions[1]);
                if (n == 1) {
                   return;
                }
              }
              String format = "png";
              if (filename.toLowerCase().endsWith(".jpg") || filename.toLowerCase().endsWith(".jpeg")) format = "jpeg";
	      UFRSSPlot upp = UFRSSPlot.this;
              try {
                BufferedImage image = new BufferedImage(upp.xdim, upp.ydim, BufferedImage.TYPE_INT_BGR);
                image.createGraphics().drawImage(offscreenImg,0,0,upp.xdim,upp.ydim,upp);
                ImageIO.write(image, format, f);
              } catch(IOException e) {
                System.err.println("UFPlotPanel error > could not create JPEG image!");
              }
           }
        }
      });

      saveItem = new JMenuItem("Save as .csv");
      popMenu.add(saveItem);
      saveItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ev) {
           JFileChooser jfc = new JFileChooser(saveDir);
           int returnVal = jfc.showSaveDialog((Component)ev.getSource());
           if (returnVal == JFileChooser.APPROVE_OPTION) {
              String filename = jfc.getSelectedFile().getAbsolutePath();
              saveDir = jfc.getCurrentDirectory();
              File f = new File(filename);
              if (f.exists()) {
                String[] saveOptions = {"Overwrite","Cancel"};
                int n = JOptionPane.showOptionDialog(UFRSSPlot.this, filename+" already exists.", "File exists!", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE, null, saveOptions, saveOptions[1]);
                if (n == 1) {
                   return;
                }
              }
	      try {
		PrintWriter pw = new PrintWriter(new FileOutputStream(f));
		pw.println(title);
		String s = "";
		int np = 0;
		boolean printWavelengths = true;
		for (int j = 0; j < numSpectra; j++) {
		  if (j > 0) s+= ", ";
		  if (printWavelengths && spectra.get(j).hasWavelength) s += "Wavelength "+j; else if (j == 0) {
		    s += "Pixel";
		    printWavelengths = false;
		  }
		  s += spectra.get(j).spectrumName;
		  np = Math.max(np, spectra.get(j).y.length);
		}
		pw.println(s);
		
		for (int j = 0; j < np; j++) {
		  if (printWavelengths) s = ""; else s = String.valueOf(j);
		  for (int l = 0; l < numSpectra; l++) {
		    if (printWavelengths) {
		      if (j < spectra.get(l).x.length) s+= ", "+String.valueOf(spectra.get(l).x[j]); else s+=", ";
		    }
		    if (j < spectra.get(l).y.length) s+= ", "+String.valueOf(spectra.get(l).y[j]); else s+=", ";
		  }
		  pw.println(s);
		}
		pw.close();
              } catch(IOException e) {
                System.err.println("UFPlotPanel error > could not create write csv file "+filename+"!");
              }
	   }
        }
      });

      printItem = new JMenuItem("Print or Save");
      popMenu.add(printItem);
      printItem.addActionListener(this);

      resetRangeItem = new JMenuItem("Reset Range");
      resetRangeItem.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent ev) {
	   plotPanel.xMinField.setText("0");
           plotPanel.yMinField.setText("");
           plotPanel.xMaxField.setText("");
           plotPanel.yMaxField.setText("");
           plotPanel.plotButton.doClick();
        }
      });
      popMenu.add(resetRangeItem);
      if (plotPanel == null) resetRangeItem.setVisible(false);

      //menu option to reset size:
      resetSizeItem = new JMenuItem("Reset Size");
      popMenu.add(resetSizeItem);
      resetSizeItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ev) {
           resizePlot(ufpxdim, ufpydim);
           if (plotPanel.getClass().getName().indexOf("SuperFatboyPlot") != -1) {
	      plotPanel.updateSize(ufpxdim+337, ufpydim+183);
           }
           else plotPanel.setSize(ufpxdim + 4, ufpydim + 26);
        }
      });

      addMouseListener(new MouseListener() {
        public void mouseClicked(MouseEvent evt) {
        }

        public void mousePressed(MouseEvent ev) {
	   for (int j = 0; j < 4; j++) dragObject[j] = false;
           if ((ev.getModifiers() & InputEvent.BUTTON3_MASK) != 0) {
              if(ev.isPopupTrigger()) {
                popMenu.show(ev.getComponent(), ev.getX(), ev.getY());
              }
           }

           if ((ev.getModifiers() & InputEvent.BUTTON1_MASK) != 0) {
              if (xyFrame != null) {
                xyFrame.xField.setText(""+ev.getX());
                xyFrame.yField.setText(""+ev.getY());
                xyFrame.deviceButton.setSelected(true);
              }
           }
	   if ((ev.getModifiers() & InputEvent.BUTTON2_MASK) != 0 ) {
	      mouseButton = 2;
	      xInit = ev.getX();
	      yInit = ev.getY();
	      for (int j = 0; j < 4; j++) {
		if (dragObjectStatus[j].equals("V Line") && Math.abs(xInit-dragObjectX[j]) < 30) {
		   xisav = dragObjectX[j];
                   yisav = 0;
                   for (int l = 0; l < 4; l++) {
		      if (l == j) dragObject[l] = true; else dragObject[l] = false;
		   }
		} else if (dragObjectStatus[j].equals("H Line") && Math.abs(yInit-dragObjectY[j]) < 30) {
		   xisav = 0;
                   yisav = dragObjectY[j];
                   for (int l = 0; l < 4; l++) {
		      if (l == j) dragObject[l] = true; else dragObject[l] = false;
                   }
                } else if (dragObjectStatus[j].equals("+") && Math.abs(xInit-dragObjectX[j]) < 30 && Math.abs(yInit-dragObjectY[j]) < 30) {
                   xisav = dragObjectX[j];
                   yisav = dragObjectY[j];
                   for (int l = 0; l < 4; l++) {
		      if (l == j) dragObject[l] = true; else dragObject[l] = false;
                   }
                } else if (dragObjectStatus[j].equals("X") && Math.abs(xInit-dragObjectX[j]) < 30 && Math.abs(yInit-dragObjectY[j]) < 30) {
                   xisav = dragObjectX[j];
                   yisav = dragObjectY[j];
                   for (int l = 0; l < 4; l++) {
	 	      if (l == j) dragObject[l] = true; else dragObject[l] = false;
                   }
                } else if (dragObjectStatus[j].equals("Circle")  && Math.abs(xInit-dragObjectX[j]) < 30 && Math.abs(yInit-dragObjectY[j]) < 30) {
                   xisav = dragObjectX[j];
                   yisav = dragObjectY[j];
                   for (int l = 0; l < 4; l++) {
		      if (l == j) dragObject[l] = true; else dragObject[l] = false;
                   }
                } else dragObject[j] = false;
	      }
	   }
        }

        public void mouseReleased(MouseEvent evt) {
        }

        public void mouseEntered(MouseEvent evt) {
        }

        public void mouseExited(MouseEvent evt) {
        }
      });

      addMouseMotionListener(new MouseMotionAdapter() {
	public void mouseMoved(MouseEvent mev) {
	   float xc = (float)(mev.getX()-xoff)/xscale;
	   float yc = (float)(yoff-mev.getY())/yscale;
	   xc = (float)(Math.floor(xc*100)*0.01);
	   yc = (float)(Math.floor(yc*100)*0.01);
           plotPanel.updateCoordLabel("["+xc+", "+yc+"]");
        }

        public void mouseDragged(MouseEvent mev) {
           float xc = (float)(mev.getX()-xoff)/xscale;
           float yc = (float)(yoff-mev.getY())/yscale;
           xc = (float)(Math.floor(xc*100)*0.01);
           yc = (float)(Math.floor(yc*100)*0.01);
           plotPanel.updateCoordLabel("["+xc+", "+yc+"]");

	   int xOffset = mev.getX()-xInit;
	   int yOffset = mev.getY()-yInit;
	   int xi = xisav + xOffset;
	   int yi = yisav + yOffset;
	   if (mouseButton == 2 ) {
	      for (int j = 0; j < 4; j++) {
		if (dragObject[j]) {
		   dragObjectX[j] = xi;
		   dragObjectY[j] = yi;
	        }
	      }
	      repaint();
	   }
        }
      });
    }

   public void initPlot() {
      offscreenImg = createImage(xdim,ydim);
      if( offscreenImg == null ) {
          plotPanel.setVisible(true);
          offscreenImg = createImage(xdim,ydim);
      }
      offscreenG = (Graphics2D)offscreenImg.getGraphics();
      offscreenG.setColor(backColor);
      offscreenG.fillRect(0,0,xdim,ydim);
      offscreenG.setColor(plotColor);
      offscreenG.setFont(mainFont);
   }
/*
    public void initPlot() {
      offscreenImg = createImage(xdim,ydim);
      while( offscreenImg == null ) {
        offscreenImg = createImage(xdim,ydim);
	try {
	  Thread.sleep(50);
	} catch(InterruptedException e) {}
      }
      offscreenG = (Graphics2D)offscreenImg.getGraphics();
      offscreenG.setColor(backColor);
      offscreenG.fillRect(0,0,xdim,ydim);
      offscreenG.setColor(plotColor);
      offscreenG.setFont(mainFont);
    }
*/

    public void parseParams(String s) {
      super.parseParams(s);
      if (xrange[1] - xrange[0] > 20) {
	for (int j = 0; j < xtickname.length; j++) {
	   xtickname[j] = ""+(int)(Float.parseFloat(xtickname[j]));
	}
      }
      if (yrange[1] - yrange[0] > 20 && !ylog) {
	for (int j = 0; j < ytickname.length; j++) {
	   ytickname[j] = ""+(int)(Float.parseFloat(ytickname[j]));
	}
      }
    }

/*
    public void setLinLog(float[] x, float[] y) {
      linX = new float[x.length];
      logX = new float[x.length];
      linY = new float[y.length];
      logY = new float[y.length];
      for (int j = 0; j < x.length; j++) {
        linX[j] = x[j];
      }
      for (int j = 0; j < y.length; j++) {
        linY[j] = y[j];
      }
    }
*/

    public void updatePlot() {
      long t = System.currentTimeMillis();
      if (numSpectra == 0) {
	float[] tempx = {0, 1};
	float[] tempy = {0, 1};
        super.plot(tempx, tempy, "*nodata");
        xyouts(0.5f, 0.5f, "No Internal Data", "*normal");
	return;
      }

      x = UFArrayOps.extractValues(spectra.get(0).x, 0, spectra.get(0).x.length-1);
      int np = x.length;
      tempy = new float[np];
      for (int j = 0; j < spectra.size(); j++) {
	x[0] = Math.min(x[0], spectra.get(j).getMinX());
	x[np-1] = Math.max(x[np-1], spectra.get(j).getMaxX());
	tempy[0] = Math.min(tempy[0], spectra.get(j).getMinY());
	if (plotOpts.indexOf("*ylog") != -1 || addOpts.indexOf("*ylog") != -1) {
	  //log plot
          if (tempy[0] <= 0) tempy[0] = spectra.get(j).getMinNonzeroY(); 
	}
	tempy[np-1] = Math.max(tempy[np-1], spectra.get(j).getMaxY());
      }
      for (int j = 1; j < tempy.length-1; j++) {
        tempy[j] = tempy[0];
      }

//System.out.println(tempy[0]+" "+tempy[np-1]+" "+x[0]+" "+x[np-1]);

      plotOpts+=",*title="+title;
      if (spectra.get(0).hasWavelength) {
	plotOpts += ",*xtitle=Wavelength";
      } else {
	plotOpts+=",*xtitle=Pixel";
      }
//System.out.println("time: "+(System.currentTimeMillis()-t));
      super.plot(x, tempy, "*nodata,"+plotOpts+addOpts);
      for (int j = 0; j < spectra.size(); j++) {
	SuperFatboySpectrum spec = spectra.get(j);
	if (spec.showSpectrum) super.overplot(spec.x, spec.y, spec.oplotOpts);
      }
      x = null;
      tempy = null;
    }

    public void calcZoom() {
      if (sxinit==0 || syinit==0 || sxfin==0 || syfin==0 ) return;
      if (Math.abs(sxinit-sxfin) < 3 && Math.abs(syinit-syfin) < 3) return;
      float x1 = (Math.min( sxinit, sxfin )-xoff)/xscale;
      float y1 = (Math.min( yoff-syinit, yoff-syfin ))/yscale;
      float x2 = (Math.max( sxinit, sxfin )-xoff)/xscale;
      float y2 = (Math.max( yoff-syinit, yoff-syfin ))/yscale;
      x1 = (float)(Math.floor(x1*1000)*.001);
      x2 = (float)(Math.floor(x2*1000)*.001);
      y1 = (float)(Math.floor(y1*1000)*.001);
      y2 = (float)(Math.floor(y2*1000)*.001);
      if (ylog) {
	y1 = (float)Math.pow(10, y1);
	y2 = (float)Math.pow(10, y2);
      }
      if ((""+x1).equals("NaN") || (""+y1).equals("NaN") || (""+x2).equals("NaN") || (""+y2).equals("NaN")) return;
      //String s = "*xrange=["+(int)x1+","+(int)x2+"], *yrange=["+(int)y1+","+(int)y2+"],"+plotOpts;
      if (plotPanel != null) {
        plotPanel.xMinField.setText(""+x1);
        plotPanel.xMaxField.setText(""+x2);
        plotPanel.yMinField.setText(""+y1);
        plotPanel.yMaxField.setText(""+y2);
        plotPanel.plotButton.doClick();
      }
    }

   public void paintComponent( Graphics g ) {
      super.paintComponent(g);
      int xlines = 0;
      int ylines = 0;
      float x1p = 0,x2p,y1p = 0,y2p;
      if (plotPanel == null) return;
      for (int j = 0; j < 4; j++) {
        g.setColor(dragObjectColor[j]);
        if (dragObjectStatus[j].equals("V Line")) {
           g.drawLine(dragObjectX[j], 0, dragObjectX[j], ydim);
           xlines++;
           if (xlines == 1) {
              plotPanel.x1l.setForeground(dragObjectColor[j]);
              x1p = (dragObjectX[j]-xoff)/xscale;
              x1p = (float)(Math.floor(x1p*1000)*.001);
              plotPanel.x1l.setText("X1="+x1p);
           } else if (xlines == 2) {
              plotPanel.x2l.setForeground(dragObjectColor[j]);
              x2p = (dragObjectX[j]-xoff)/xscale;
              x2p = (float)(Math.floor(x2p*1000)*.001);
              plotPanel.x2l.setText("X2="+x2p);
              plotPanel.dxl.setText("dX="+(x1p-x2p));
           }
        } else if (dragObjectStatus[j].equals("H Line")) {
           g.drawLine(0, dragObjectY[j], xdim, dragObjectY[j]);
           ylines++;
           if (ylines == 1) {
              plotPanel.y1l.setForeground(dragObjectColor[j]);
              y1p = (yoff-dragObjectY[j])/yscale;
              y1p = (float)(Math.floor(y1p*1000)*.001);
              plotPanel.y1l.setText("Y1="+y1p);
           } else if (ylines == 2) {
              plotPanel.y2l.setForeground(dragObjectColor[j]);
              y2p = (yoff-dragObjectY[j])/yscale;
              y2p = (float)(Math.floor(y2p*1000)*.001);
              plotPanel.y2l.setText("Y2="+y2p);
              plotPanel.dyl.setText("dY="+(y1p-y2p));
           }
        } else if (dragObjectStatus[j].equals("+")) {
           int xc = dragObjectX[j];
           int yc = dragObjectY[j];
           g.drawLine(xc,yc-20,xc,yc+20);
           g.drawLine(xc-20,yc,xc+20,yc);
           g.setColor(Color.BLACK);
           g.drawLine(xc-1,yc-20,xc-1,yc+20);
           g.drawLine(xc+1,yc-20,xc+1,yc+20);
           g.drawLine(xc-20,yc-1,xc+20,yc-1);
           g.drawLine(xc-20,yc+1,xc+20,yc+1);
           g.setColor(dragObjectColor[j]);
        } else if (dragObjectStatus[j].equals("X")) {
           int xc = dragObjectX[j];
           int yc = dragObjectY[j];
           g.drawLine(xc-20,yc-20,xc+20,yc+20);
           g.drawLine(xc+20,yc-20,xc-20,yc+20);
           g.setColor(Color.BLACK);
           g.drawLine(xc-21,yc-20,xc+19,yc+20);
           g.drawLine(xc-19,yc-20,xc+21,yc+20);
           g.drawLine(xc+21,yc-20,xc-19,yc+20);
           g.drawLine(xc+19,yc-20,xc-21,yc+20);
           g.setColor(dragObjectColor[j]);
        } else if (dragObjectStatus[j].equals("Circle")) {
           g.drawOval(dragObjectX[j]-10, dragObjectY[j]-10, 20, 20);
           g.setColor(Color.BLACK);
           g.drawOval(dragObjectX[j]-11, dragObjectY[j]-11, 22, 22);
           g.drawOval(dragObjectX[j]-9, dragObjectY[j]-9, 18, 18);
           g.setColor(dragObjectColor[j]);
        }
      }
      if (xlines < 2) {
        plotPanel.x2l.setForeground(Color.BLACK);
        plotPanel.x2l.setText("X2=");
        plotPanel.dxl.setText("");
      }
      if (xlines < 1) {
        plotPanel.x1l.setForeground(Color.BLACK);
        plotPanel.x1l.setText("X1=");
      }
      if (ylines < 2) {
        plotPanel.y2l.setForeground(Color.BLACK);
        plotPanel.y2l.setText("Y2=");
        plotPanel.dyl.setText("");
      }
      if (ylines < 1) {
        plotPanel.y1l.setForeground(Color.BLACK);
        plotPanel.y1l.setText("Y1=");
      }
    }

//-----------------------------------------------------------------------------//

    public static UFFloats castAsUFFloats (UFProtocol frameData) {
        if (frameData instanceof UFFloats) {
           return (UFFloats)frameData;
        } else if (frameData instanceof UFShorts) {
           return new UFFloats(frameData.name(), UFArrayOps.castAsFloats(((UFShorts)frameData).values()));
        } else if (frameData instanceof UFInts) {
           return new UFFloats(frameData.name(), UFArrayOps.castAsFloats(((UFInts)frameData).values()));
        }
        System.err.println("Invalid data type specified");
        return null;
    }

//-----------------------------------------------------------------------------//

    public static void main(String[] args) {
      JFrame jf = new JFrame();
      UFRSSPlot hp = new UFRSSPlot();
      jf.getContentPane().add(hp);
      jf.pack();
      jf.setVisible(true);
    }

// ================================== class RSSContainer ===================//

    public class RSSContainer {
      //helper class containing label, etc.
      public String label;
      public int position, numSpec;
      public JLabel fileLabel;
      public JCheckBox plotAllCheckbox;
      public UFColorButton deleteButton;
      public UFRSSPlot thePlot;

      public RSSContainer(String lab, int p, int n, UFRSSPlot plot) {
	label = lab;
	if (label.lastIndexOf("/") != -1) label = label.substring(label.lastIndexOf("/")+1);
	position = p;
	numSpec = n;
	thePlot = plot;

        fileLabel = new JLabel(label);
        fileLabel.setBorder(new EtchedBorder());
        plotAllCheckbox = new JCheckBox();
	plotAllCheckbox.setSelected(true);

	plotAllCheckbox.addActionListener(new ActionListener() {
	  public void actionPerformed(ActionEvent ev) {
	    for (int j = position; j < position+numSpec; j++) {
	      thePlot.spectra.get(j).plotCheckbox.setSelected(plotAllCheckbox.isSelected());
	      thePlot.spectra.get(j).showSpectrum = plotAllCheckbox.isSelected();
	    }
	    if (!thePlot.plotPanel.multiMode) {
	      thePlot.plotPanel.startPlot();
	      thePlot.updatePlot();
	    }
	  }
	});

	deleteButton = new UFColorButton("X", UFColorButton.COLOR_SCHEME_RED);
	deleteButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ev) {
	    for (int j = position+numSpec-1; j >= position; j--) {
	      SuperFatboySpectrum spec = thePlot.spectra.get(j);
	      thePlot.spectra.remove(spec); 
	    }
	    thePlot.numSpectra -= numSpec;
	    int idx = thePlot.rssContainers.indexOf(RSSContainer.this);
	    for (int j = idx+1; j < rssContainers.size(); j++) {
	      rssContainers.get(j).position -= numSpec;
	    }
	    thePlot.rssContainers.remove(RSSContainer.this);
	    thePlot.plotPanel.setupLeftPanel();
	    thePlot.plotPanel.revalidate();
	    thePlot.plotPanel.repaint();
	    if (!thePlot.plotPanel.multiMode) {
	      thePlot.plotPanel.startPlot();
              thePlot.updatePlot();
	    }
          }
        });
      }
    }


// ================================== class SuperFatboySpectrum ===================//

    public class SuperFatboySpectrum {
      public JTextField colorField, symField;
      public JCheckBox plotCheckbox;
      public String spectrumName, fitsFilename, oplotOpts;
      public int spectrumNumber;
      public boolean showSpectrum, hasWavelength;

      public float[] x, y;

      public SuperFatboySpectrum(UFFITSheader fitsHead, float[] ys, int n) {
        spectrumNumber = n;
	fitsFilename = fitsHead.getFilename();
	if (fitsFilename.lastIndexOf("/") != -1) fitsFilename = fitsFilename.substring(fitsFilename.lastIndexOf("/")+1);

	showSpectrum = true;
	hasWavelength = true;
	spectrumName = "Spectrum "+n;
        oplotOpts = "";
        int cols = fitsHead.getLastFrameCon().width;
	y = UFArrayOps.extractValues(ys, n*cols, (n+1)*cols-1);

	String slitStr = String.valueOf(n+1); 
	if (n+1 < 10) slitStr = "0"+slitStr;
	if (fitsHead.hasRecord("SPEC_"+slitStr)) {
	  //has SPEC_xx keyword which identifies slitlet
	  //Possible to have more than one spectrum per slitlet
	  String specStr = fitsHead.getRecord("SPEC_"+slitStr);
	  slitStr = specStr.substring(specStr.indexOf("Slitlet")+8, specStr.indexOf(":"));
	  if (slitStr.length() == 1) slitStr = "0"+slitStr;
	}

	x = new float[y.length];
        if (fitsHead.hasRecord("HIERARCH PORDER"+slitStr+"_SEG0")) {
	  int nseg = fitsHead.getInt("NSEG_"+slitStr);
	  int stride = y.length/nseg;
          //Individual wavelength solution for this segment 
          float[] xs = new float[stride+1];
          float[] segws = new float[stride+1];
          //Calculate wavelength solution
          //Use PORDERxx_SEGy and PCFi_Sxx_SEGy
	  for (int seg = 0; seg < nseg; seg++) {
	    for (int j = 0; j < xs.length; j++) xs[j] = j;
	    int order = fitsHead.getInt("HIERARCH PORDER"+slitStr+"_SEG"+seg);
	    float offset = fitsHead.getFloat("HIERARCH PCF0_S"+slitStr+"_SEG"+seg);
	    for (int j = 0; j < segws.length; j++) segws[j] = offset;
            if (order >= 1) segws = UFArrayOps.addArrays(segws, UFArrayOps.multArrays(xs, fitsHead.getFloat("HIERARCH PCF1_S"+slitStr+"_SEG"+seg)));
            for (int j = 2; j <= order; j++) {
              segws = UFArrayOps.addArrays(segws, UFArrayOps.multArrays(UFArrayOps.powArray(xs, (float)j), fitsHead.getFloat("HIERARCH PCF"+j+"_S"+slitStr+"_SEG"+seg)));
            }
	    //update overall x axis array
	    for (int j = 0; j < stride+1 && j+seg*stride < x.length; j++) x[j+seg*stride] = segws[j];
	    if (seg != 0) y[seg*stride] = 0;
	    y[(seg+1)*stride-1] = 0;
	  }
	} else if (fitsHead.hasRecord("PORDER"+slitStr)) {
          //Individual wavelength solution for this slitlet
          float[] xs = new float[y.length];
          for (int j = 0; j < xs.length; j++) xs[j] = j;
          //Calculate wavelength solution
          //Use PORDERxx and PCFi_Sxx
          int order = fitsHead.getInt("PORDER"+slitStr);
          float offset = fitsHead.getFloat("PCF0_S"+slitStr);
          for (int j = 0; j < x.length; j++) x[j] = offset;
          if (order >= 1) x = UFArrayOps.addArrays(x, UFArrayOps.multArrays(xs, fitsHead.getFloat("PCF1_S"+slitStr)));
          for (int j = 2; j <= order; j++) {
            x = UFArrayOps.addArrays(x, UFArrayOps.multArrays(UFArrayOps.powArray(xs, (float)j), fitsHead.getFloat("PCF"+j+"_S"+slitStr)));
          }
        } else if (fitsHead.hasRecord("PORDER")) {
          //One single wavelength solution
          float[] xs = new float[y.length];
	  for (int j = 0; j < xs.length; j++) xs[j] = j;
          //Calculate wavelength solution
          int order = fitsHead.getInt("PORDER");
          float offset = fitsHead.getFloat("PCOEFF_0");
	  for (int j = 0; j < x.length; j++) x[j] = offset;
	  if (order >= 1) x = UFArrayOps.addArrays(x, UFArrayOps.multArrays(xs, fitsHead.getFloat("PCOEFF_1"))); 
          for (int j = 2; j <= order; j++) { 
	    x = UFArrayOps.addArrays(x, UFArrayOps.multArrays(UFArrayOps.powArray(xs, (float)j), fitsHead.getFloat("PCOEFF_"+j)));
	  }
	} else if (fitsHead.hasRecord("CRVAL1") && fitsHead.hasRecord("CDELT1")) {
	  //Linear wavelength solution
	  float offset = fitsHead.getFloat("CRVAL1");
	  float delta = fitsHead.getFloat("CDELT1");
	  for (int j = 0; j < x.length; j++) x[j] = offset+delta*j;
	} else {
	  for (int j = 0; j < x.length; j++) x[j] = j;
	  hasWavelength = false;
	}

	colorField = new JTextField(8);
	symField = new JTextField(2);
	plotCheckbox = new JCheckBox(spectrumName, true);
	plotCheckbox.addActionListener(new ActionListener() {
	  public void actionPerformed(ActionEvent ev) {
	    JCheckBox temp = (JCheckBox)ev.getSource();
	    showSpectrum = temp.isSelected();
	    if (!plotPanel.multiMode) { 
	      plotPanel.startPlot();
              updatePlot();
	    }
          }
        });
      }

      public float getMaxX() { return UFArrayOps.maxValue(x); }
      public float getMinX() { return UFArrayOps.minValue(x); }
      public float getMaxY() { return UFArrayOps.maxValue(y); }
      public float getMinY() { return UFArrayOps.minValue(y); }
      public float getMinNonzeroY() { return UFArrayOps.minValue(UFArrayOps.extractValues(y, UFArrayOps.where(y, ">", 0))); }
    }

}
